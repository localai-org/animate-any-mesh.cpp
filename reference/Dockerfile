ARG PYTORCH_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ARG UPSTREAM_REVISION=a0e1d824e59ff4725799cb70511cdd55b30d750b

RUN python -m pip install --no-cache-dir \
      numpy==1.26.0 \
      torchdiffeq==0.2.5 \
      transformers==4.55.3 \
      einops==0.8.1 \
      trimesh==4.7.4

ENV PYTHONPATH=/opt/AnimateAnyMesh
ENV TOKENIZERS_PARALLELISM=false
WORKDIR /work

LABEL org.opencontainers.image.source="https://github.com/JarrentWu1031/AnimateAnyMesh" \
      org.opencontainers.image.revision="$UPSTREAM_REVISION"

ENTRYPOINT ["python"]
