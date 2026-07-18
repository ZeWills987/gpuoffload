# gpuoffload

Délégation à la demande des calculs neuronaux d'audiotwin vers un GPU
loué, **piloté par votre serveur de base** : le pod est créé quand un
job arrive, les fichiers audio voyagent dans la requête, les résultats
reviennent en JSON au serveur de base, et le pod est **détruit après un
délai d'inactivité** — vous ne payez que les secondes utiles.

```
serveur de base (mkzik-api-py)                pod GPU éphémère (RunPod)
┌─────────────────────────┐                  ┌──────────────────────────┐
│ orchestrator.GPUOffload │── crée le pod ──▶│ executor.py (HTTP + token)│
│  - queue implicite      │── POST /run ────▶│  audiotwin.neural (cuda) │
│  - données conservées   │◀─── JSON ────────│  aucun stockage          │
│  - idle timeout         │── DELETE pod ───▶│  (détruit)               │
└─────────────────────────┘                  └──────────────────────────┘
```

## Test local (gratuit, valide tout le flux)

```bash
python orchestrator.py health --provider local
python orchestrator.py run neural_similarity a.mp3 b.mp3 --provider local
```

Le provider `local` lance `executor.py` en sous-processus (CPU) : même
protocole, même code que le vrai pod.

## Mode RunPod (GPU loué à la demande)

Prérequis (une fois) : compte sur runpod.com + crédit, puis une clé API
(Settings → API Keys).

```bash
export RUNPOD_API_KEY="..."

python orchestrator.py run neural_similarity a.mp3 b.mp3 --provider runpod
```

Par défaut, le provider `runpod` lance l'**image pré-construite**
`ghcr.io/<owner>/gpuoffload-executor:latest` (voir `Dockerfile` +
`.github/workflows/publish-executor-image.yml`, rebuild automatique à
chaque push sur `master`) — audiotwin, sampleid et le checkpoint sont
déjà installés dedans, le pod n'a plus qu'à démarrer `executor.py` :
boot en quelques secondes au lieu de 3-6 min.

Mode legacy (install au boot, plus lent, utile si l'image n'est pas
encore publiée) : passez une image "brute" et l'URL de `executor.py` —

```bash
export GPUOFFLOAD_EXECUTOR_URL="https://raw.githubusercontent.com/<vous>/gpuoffload/master/executor.py"
python -c "
from orchestrator import GPUOffload
off = GPUOffload(provider='runpod', image='runpod/pytorch:1.0.7-cu1281-torch280-ubuntu2404')
"
```

Ou en bibliothèque, depuis votre code serveur :

```python
from orchestrator import GPUOffload

off = GPUOffload(provider="runpod", idle_timeout=600)
nfp = off.run("neural_similarity", "/data/a.mp3", "/data/b.mp3")
loc = off.run("neural_localized_match", "/data/remix.mp3", "/data/original.mp3")
# ... enchaînez les jobs : le pod reste chaud entre deux,
# et s'autodétruit 600 s après le dernier.
```

Tâches disponibles : `neural_similarity`, `neural_match_points`,
`neural_localized_match`, `neural_embedding` (+ `kwargs` passés tels
quels aux fonctions audiotwin).

## Coûts et sécurité

- **Premier boot du pod (image prébuilt) : quelques secondes.** Avec
  l'image legacy (install au boot) : ~3-6 min (apt + pip + checkpoint
  805 Mo) — l'essentiel de ce temps reste facturé par RunPod même s'il
  ne calcule rien (téléchargement de la grosse image PyTorch/CUDA de
  base par leur infra). Jobs suivants sur un pod déjà chaud : ~1-3 s
  par paire.
- **Idle timeout** (600 s par défaut) : le filet anti-facturation. En
  cas d'échec du DELETE, un avertissement explicite vous renvoie vers
  la console RunPod.
- **Token bearer** généré par lancement (secrets.token_urlsafe) et
  injecté dans le pod : personne d'autre ne peut soumettre de jobs.
  Le proxy RunPod fournit le HTTPS.
- **Aucune donnée ne persiste sur le pod** : fichiers écrits en tempdir
  par job, supprimés aussitôt, pod détruit ensuite.
- RTX 4090 ≈ 0,30-0,50 $/h facturé à la seconde d'existence du pod.

## Intégration plateforme (rappel d'architecture)

Le coût neuronal est **par track, pas par paire** : à l'échelle d'un
catalogue, précalculez les embeddings à l'ingestion
(`off.run("neural_embedding", path)` → stockez le vecteur), et faites
les comparaisons par paires en produits matriciels numpy côté serveur
de base — le GPU ne sert alors qu'à l'ingestion.
