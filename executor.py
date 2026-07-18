#!/usr/bin/env python3
"""gpuoffload executor — runs ON the GPU pod (or locally for testing).

A minimal stdlib HTTP server exposing audiotwin's neural tasks. The
orchestrator (on the base server) POSTs a job with the audio files
embedded as base64; the executor writes them to a temp dir, runs the
requested audiotwin function on the local device (GPU if available),
and returns the JSON result. No data is stored on the pod: temp files
are deleted after each job.

Security: every request must carry ``Authorization: Bearer <token>``,
where the token is injected by the orchestrator at pod-launch time
(``GPUOFFLOAD_TOKEN`` env var or --token).

Usage:
    python executor.py --port 8000 --token SECRET
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import tempfile
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_BODY_BYTES = 200 * 1024 * 1024  # 200 MB: two long tracks in base64

TOKEN = os.environ.get("GPUOFFLOAD_TOKEN", "")


def _tasks() -> dict:
    """Whitelist of executable tasks (lazy imports keep startup instant)."""
    from audiotwin.neural import (
        neural_embedding,
        neural_localized_match,
        neural_match_points,
        neural_similarity,
    )
    from audiotwin.scores import _neural_scores

    return {
        "neural_similarity": {"fn": neural_similarity, "files": 2},
        "neural_match_points": {"fn": neural_match_points, "files": 2},
        "neural_localized_match": {"fn": neural_localized_match, "files": 2},
        "neural_embedding": {"fn": neural_embedding, "files": 1},
        # Miroir exact de audiotwin.scores._neural_scores (mêmes clés :
        # neural_similarity, neural_similarity_raw) — c'est ce que
        # mkzik-mir-poc consomme via pipeline.extract_features().
        "neural_scores": {"fn": _neural_scores, "files": 2},
    }


def _run_task(payload: dict) -> dict:
    tasks = _tasks()
    name = payload.get("task")
    if name not in tasks:
        raise ValueError(f"unknown task {name!r}; allowed: {sorted(tasks)}")
    spec = tasks[name]

    files = payload.get("files") or {}
    kwargs = payload.get("kwargs") or {}
    expected = ["a", "b"][: spec["files"]]
    missing = [k for k in expected if k not in files]
    if missing:
        raise ValueError(f"task {name!r} needs files {expected}, missing {missing}")

    with tempfile.TemporaryDirectory(prefix="gpuoffload_") as tmp:
        paths = []
        for key in expected:
            entry = files[key]
            suffix = Path(entry.get("name", f"{key}.mp3")).suffix or ".mp3"
            path = os.path.join(tmp, f"{key}{suffix}")
            with open(path, "wb") as f:
                f.write(base64.b64decode(entry["data_b64"]))
            paths.append(path)

        result = spec["fn"](*paths, **kwargs)

    # numpy arrays (neural_embedding) -> lists for JSON
    if hasattr(result, "tolist"):
        result = {"embedding": result.tolist()}
    return result


class Handler(BaseHTTPRequestHandler):
    server_version = "gpuoffload/1.0"

    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        return TOKEN and self.headers.get("Authorization") == f"Bearer {TOKEN}"

    def do_GET(self):  # noqa: N802 (stdlib naming)
        if self.path != "/health":
            return self._send(404, {"error": "not found"})
        if not self._authorized():
            return self._send(401, {"error": "unauthorized"})
        info = {"status": "ok"}
        try:
            import torch

            import audiotwin
            from audiotwin.neural import _resolve_device

            info.update(
                device=_resolve_device(),
                cuda_available=torch.cuda.is_available(),
                audiotwin=audiotwin.__version__,
                torch=torch.__version__,
            )
        except Exception as exc:  # noqa: BLE001 — health must answer
            info.update(status="degraded", error=f"{type(exc).__name__}: {exc}")
        self._send(200, info)

    def do_POST(self):  # noqa: N802
        if self.path != "/run":
            return self._send(404, {"error": "not found"})
        if not self._authorized():
            return self._send(401, {"error": "unauthorized"})
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > MAX_BODY_BYTES:
            return self._send(413, {"error": f"body size {length} out of bounds"})
        try:
            payload = json.loads(self.rfile.read(length))
            result = _run_task(payload)
            self._send(200, {"ok": True, "result": result})
        except Exception as exc:  # noqa: BLE001 — report, don't crash the pod
            self._send(
                500,
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=5),
                },
            )

    def log_message(self, fmt, *args):  # quieter logs, no client data
        print(f"[executor] {self.address_string()} {fmt % args}")


def main() -> None:
    global TOKEN
    parser = argparse.ArgumentParser(description="gpuoffload executor")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--token", default=TOKEN, help="bearer token (or GPUOFFLOAD_TOKEN env)")
    args = parser.parse_args()

    if not args.token:
        raise SystemExit("a token is required (--token or GPUOFFLOAD_TOKEN)")
    TOKEN = args.token

    # Warm up the model at boot so the first job doesn't pay the load.
    try:
        from audiotwin.neural import _load_model, _resolve_device

        print(f"[executor] warming model on {_resolve_device()}...")
        _load_model()
        print("[executor] model ready")
    except Exception as exc:  # noqa: BLE001
        print(f"[executor] warmup failed (jobs will retry): {type(exc).__name__}: {exc}")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[executor] listening on {args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
