# Image executor pre-construite : audiotwin + sampleid + checkpoint deja
# installes, pour que le pod RunPod demarre en secondes plutot qu'en
# minutes (plus de git clone / pip install / download au boot).
#
# Build & push automatises par .github/workflows/publish-executor-image.yml
# (image publique sur ghcr.io/<owner>/gpuoffload-executor).
#
# Build manuel :
#   docker build -t gpuoffload-executor .
#   docker run --gpus all -e GPUOFFLOAD_TOKEN=test -p 8000:8000 gpuoffload-executor

FROM runpod/pytorch:1.0.7-cu1281-torch280-ubuntu2404

RUN apt-get update -qq \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
        ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN pip install --no-cache-dir "audiotwin[all] @ git+https://github.com/ZeWills987/audiotwin.git"
RUN pip install --no-cache-dir -e "git+https://github.com/sony/sampleid.git#egg=sampleid"

# Precharge le checkpoint Zenodo (805 Mo) dans l'image : aucun
# telechargement au demarrage du pod.
RUN python -c "from audiotwin.neural import _require_sampleid, _default_checkpoint; \
    _default_checkpoint(_require_sampleid())"

COPY executor.py /executor.py

EXPOSE 8000
ENTRYPOINT ["python", "/executor.py", "--port", "8000"]
