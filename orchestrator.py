#!/usr/bin/env python3
"""gpuoffload orchestrator — runs on the BASE server.

Launches an ephemeral GPU executor when work arrives, forwards jobs
(audio files travel inside the request, results come back as JSON so
all data stays under the base server's control), and tears the executor
down after an idle timeout to minimize rental costs.

Providers:
    local   — spawns executor.py as a subprocess on this machine
              (CPU; free; for testing the full flow end to end)
    runpod  — creates/destroys a RunPod GPU pod via their REST API
              (needs RUNPOD_API_KEY; see README for setup)

Library usage:
    from orchestrator import GPUOffload
    off = GPUOffload(provider="local")           # or "runpod"
    nfp = off.run("neural_similarity", "a.mp3", "b.mp3")
    off.shutdown()                               # or rely on idle timeout

CLI:
    python orchestrator.py health --provider local
    python orchestrator.py run neural_similarity a.mp3 b.mp3 --provider local
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_IDLE_TIMEOUT = 600.0  # seconds without a job before teardown
DEFAULT_JOB_TIMEOUT = 900.0  # seconds per job (GPU: expect a few seconds)
BOOT_TIMEOUT = 900.0  # seconds to wait for the executor to come up

RUNPOD_API = "https://rest.runpod.io/v1"


# ============================================================
# Providers — lifecycle of the executor
# ============================================================


class LocalProvider:
    """Executor as a local subprocess — free end-to-end testing."""

    name = "local"

    def __init__(self, port: int = 8791):
        self._port = port
        self._proc: subprocess.Popen | None = None

    def launch(self, token: str) -> str:
        executor = Path(__file__).resolve().parent / "executor.py"
        env = dict(os.environ, GPUOFFLOAD_TOKEN=token)
        self._proc = subprocess.Popen(
            [sys.executable, str(executor), "--port", str(self._port), "--host", "127.0.0.1"],
            env=env,
        )
        return f"http://127.0.0.1:{self._port}"

    def terminate(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None


#: Image pre-construite (audiotwin + sampleid + checkpoint deja installes) —
#: voir Dockerfile + .github/workflows/publish-executor-image.yml. Boot en
#: quelques secondes au lieu de plusieurs minutes (plus de git clone / pip
#: install / download du checkpoint à chaque lancement de pod).
PREBUILT_IMAGE = "ghcr.io/zewills987/gpuoffload-executor:latest"


class RunPodProvider:
    """Ephemeral RunPod GPU pod via their REST API.

    Par defaut, lance PREBUILT_IMAGE (deja pretes : ffmpeg, audiotwin,
    sampleid, checkpoint) — le pod n'a plus qu'a demarrer executor.py,
    aucune installation au boot.

    Passez ``image=`` a une image "brute" (ex. runpod/pytorch:...) pour
    revenir au mode legacy install-au-boot (plus lent, utile si l'image
    prebuilt n'est pas encore publiee ou pour deboguer une regression).
    """

    name = "runpod"

    #: Types de GPU essayes dans l'ordre quand RunPod repond "This machine
    #: does not have the resources" (disponibilite fluctuante). Tous ont
    #: largement assez de VRAM pour Sample-ID (inference seule) ; l'ordre
    #: privilegie le moins cher a l'heure.
    GPU_FALLBACKS = [
        "NVIDIA RTX A4000",
        "NVIDIA GeForce RTX 3090",
        "NVIDIA GeForce RTX 4090",
    ]

    def __init__(
        self,
        api_key: str | None = None,
        executor_url: str | None = None,
        gpu_type: str | list[str] | None = None,
        image: str = PREBUILT_IMAGE,
        disk_gb: int = 20,
        pod_name: str = "gpuoffload-executor",
    ):
        self._api_key = api_key or os.environ.get("RUNPOD_API_KEY", "")
        if not self._api_key:
            raise SystemExit("RUNPOD_API_KEY manquant (env var ou api_key=)")
        self._prebuilt = image == PREBUILT_IMAGE
        self._executor_url = executor_url or os.environ.get("GPUOFFLOAD_EXECUTOR_URL", "")
        if not self._prebuilt and not self._executor_url:
            raise SystemExit(
                "URL du script executor manquante (GPUOFFLOAD_EXECUTOR_URL) — "
                "hébergez executor.py à une URL brute (repo git, gist...), ou "
                "utilisez l'image prebuilt par defaut (aucune URL requise)"
            )
        if gpu_type is None:
            self._gpu_types = list(self.GPU_FALLBACKS)
        elif isinstance(gpu_type, str):
            self._gpu_types = [gpu_type]
        else:
            self._gpu_types = list(gpu_type)
        self._image = image
        self._disk_gb = disk_gb
        self._pod_name = pod_name
        self._pod_id: str | None = None

    def _api(self, method: str, path: str, payload: dict | None = None) -> dict:
        req = urllib.request.Request(
            f"{RUNPOD_API}{path}",
            method=method,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload).encode() if payload is not None else None,
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read() or "{}")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise RuntimeError(f"RunPod API {exc.code} sur {method} {path} : {body}") from exc

    def launch(self, token: str) -> str:
        payload = {
            "name": self._pod_name,
            "imageName": self._image,
            "gpuCount": 1,
            "containerDiskInGb": self._disk_gb,
            "ports": ["8000/http"],
            "env": {
                "GPUOFFLOAD_TOKEN": token,
                "AUDIOTWIN_DEVICE": "cuda",
                # Ubuntu 24.04 : pip systeme protege par PEP 668.
                "PIP_BREAK_SYSTEM_PACKAGES": "1",
            },
        }
        if self._prebuilt:
            # Tout est deja installe dans l'image — on laisse son
            # ENTRYPOINT (python /executor.py --port 8000) demarrer seul.
            pass
        else:
            start_cmd = (
                "bash -lc '"
                "apt-get update -qq && apt-get install -y -qq ffmpeg curl git && "
                'pip install -q "audiotwin[all] @ git+https://github.com/ZeWills987/audiotwin.git" && '
                'pip install -q -e "git+https://github.com/sony/sampleid.git#egg=sampleid" && '
                f"curl -fsSL {self._executor_url} -o /executor.py && "
                "python /executor.py --port 8000'"
            )
            payload["dockerStartCmd"] = ["bash", "-c", start_cmd]

        # La disponibilite RunPod fluctue ("This machine does not have the
        # resources...") : on essaie chaque type de GPU dans l'ordre.
        last_error: Exception | None = None
        for gpu_type in self._gpu_types:
            payload["gpuTypeIds"] = [gpu_type]
            try:
                pod = self._api("POST", "/pods", payload)
            except RuntimeError as exc:
                print(f"[orchestrator] {gpu_type} indisponible : {exc}")
                last_error = exc
                continue
            self._pod_id = pod.get("id") or pod.get("podId")
            if not self._pod_id:
                raise RuntimeError(f"création du pod échouée : {pod}")
            print(f"[orchestrator] pod RunPod créé : {self._pod_id} ({gpu_type})")
            return f"https://{self._pod_id}-8000.proxy.runpod.net"
        raise RuntimeError(
            f"aucun GPU disponible parmi {self._gpu_types} — réessayez dans "
            f"quelques minutes (dernière erreur : {last_error})"
        )

    def terminate(self) -> None:
        if self._pod_id:
            try:
                self._api("DELETE", f"/pods/{self._pod_id}")
                print(f"[orchestrator] pod {self._pod_id} détruit")
            except urllib.error.URLError as exc:
                print(f"[orchestrator] ATTENTION: destruction du pod échouée ({exc}) — "
                      f"vérifiez la console RunPod pour éviter la facturation !")
            self._pod_id = None


PROVIDERS = {"local": LocalProvider, "runpod": RunPodProvider}


# ============================================================
# Orchestrator
# ============================================================


class GPUOffload:
    """Lazy executor lifecycle + job forwarding + idle teardown."""

    def __init__(
        self,
        provider: str = "local",
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
        job_timeout: float = DEFAULT_JOB_TIMEOUT,
        **provider_kwargs,
    ):
        self._provider = PROVIDERS[provider](**provider_kwargs)
        self._idle_timeout = idle_timeout
        self._job_timeout = job_timeout
        # GPUOFFLOAD_TOKEN fixe (pratique pour deboguer le pod a la main
        # avec curl) ; sinon un token aleatoire par lancement, plus sur.
        self._token = os.environ.get("GPUOFFLOAD_TOKEN") or secrets.token_urlsafe(32)
        self._base_url: str | None = None
        self._last_job = 0.0
        self._lock = threading.Lock()
        self._reaper: threading.Timer | None = None

    # --- lifecycle -------------------------------------------------

    def _ensure_up(self) -> str:
        with self._lock:
            if self._base_url is None:
                self._base_url = self._provider.launch(self._token)
                self._wait_healthy()
            return self._base_url

    def _wait_healthy(self) -> None:
        deadline = time.time() + BOOT_TIMEOUT
        delay = 2.0
        last_report = 0.0
        last_error = "?"
        while time.time() < deadline:
            try:
                health = self._request("GET", "/health", timeout=10)
                print(f"[orchestrator] executor prêt : {health}")
                return
            except urllib.error.HTTPError as exc:
                # 401 = token divergent, 403 = probablement bloque par le
                # proxy — des erreurs de CONFIG, pas de boot : reessayer
                # en boucle ne les resoudra jamais.
                body = exc.read().decode(errors="replace")[:200]
                if exc.code in (401, 403):
                    self.shutdown()
                    raise RuntimeError(
                        f"l'executor repond mais refuse la requete "
                        f"(HTTP {exc.code} : {body}) — verifiez le token / le proxy"
                    ) from exc
                last_error = f"HTTP {exc.code} : {body}"
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            # Un point d'avancement toutes les ~30 s, sinon silence radio.
            if time.time() - last_report > 30:
                remaining = deadline - time.time()
                print(f"[orchestrator] en attente de l'executor "
                      f"({remaining:.0f}s restantes) — dernier essai : {last_error}")
                last_report = time.time()
            time.sleep(delay)
            delay = min(delay * 1.5, 20.0)
        self.shutdown()
        raise RuntimeError(
            f"l'exécuteur n'a pas démarré en {BOOT_TIMEOUT:.0f}s "
            f"(dernière erreur : {last_error})"
        )

    def _touch(self) -> None:
        self._last_job = time.time()
        if self._reaper:
            self._reaper.cancel()
        self._reaper = threading.Timer(self._idle_timeout, self._reap_if_idle)
        self._reaper.daemon = True
        self._reaper.start()

    def _reap_if_idle(self) -> None:
        if time.time() - self._last_job >= self._idle_timeout:
            print(f"[orchestrator] inactif depuis {self._idle_timeout:.0f}s — teardown")
            self.shutdown()

    def shutdown(self) -> None:
        with self._lock:
            if self._reaper:
                self._reaper.cancel()
                self._reaper = None
            if self._base_url is not None:
                self._provider.terminate()
                self._base_url = None

    # --- jobs ------------------------------------------------------

    def _request(self, method: str, path: str, payload: dict | None = None, timeout=None):
        req = urllib.request.Request(
            f"{self._base_url}{path}",
            method=method,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
                # Le proxy RunPod est derriere Cloudflare, qui peut filtrer
                # le User-Agent par defaut de urllib (Python-urllib/3.x).
                "User-Agent": "gpuoffload/1.0",
            },
            data=json.dumps(payload).encode() if payload is not None else None,
        )
        with urllib.request.urlopen(req, timeout=timeout or self._job_timeout) as resp:
            return json.loads(resp.read())

    def run(self, task: str, path_a: str, path_b: str | None = None, **kwargs) -> dict:
        """Execute one task on the (lazily launched) executor.

        The audio files are read HERE and travel inside the request; the
        executor keeps nothing. Returns the task's JSON result.
        """
        self._ensure_up()
        files = {"a": self._pack(path_a)}
        if path_b is not None:
            files["b"] = self._pack(path_b)
        response = self._request(
            "POST", "/run", {"task": task, "files": files, "kwargs": kwargs}
        )
        self._touch()
        if not response.get("ok"):
            raise RuntimeError(
                f"job {task} échoué côté exécuteur : {response.get('error')}\n"
                f"{response.get('traceback', '')}"
            )
        return response["result"]

    def health(self) -> dict:
        self._ensure_up()
        return self._request("GET", "/health", timeout=30)

    @staticmethod
    def _pack(path: str) -> dict:
        data = Path(path).read_bytes()
        return {"name": Path(path).name, "data_b64": base64.b64encode(data).decode("ascii")}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()


# ============================================================
# CLI
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="gpuoffload orchestrator")
    parser.add_argument("command", choices=["health", "run"])
    parser.add_argument("task", nargs="?", help="task name (for 'run')")
    parser.add_argument("file_a", nargs="?")
    parser.add_argument("file_b", nargs="?")
    parser.add_argument("--provider", default="local", choices=sorted(PROVIDERS))
    parser.add_argument("--idle-timeout", type=float, default=DEFAULT_IDLE_TIMEOUT)
    parser.add_argument("--kwargs", default="{}", help="JSON kwargs for the task")
    args = parser.parse_args()

    with GPUOffload(provider=args.provider, idle_timeout=args.idle_timeout) as off:
        if args.command == "health":
            print(json.dumps(off.health(), indent=2))
        elif args.command == "run":
            if not args.task or not args.file_a:
                parser.error("run exige: task file_a [file_b]")
            result = off.run(args.task, args.file_a, args.file_b, **json.loads(args.kwargs))
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
