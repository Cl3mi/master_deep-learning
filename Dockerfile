# Pinned by digest so a rebuild can never pick up a different base image.
# Re-resolve with:
#   docker pull python:3.12-slim-bookworm && \
#   docker inspect --format='{{index .RepoDigests 0}}' python:3.12-slim-bookworm
FROM python:3.12-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2

# Each variable removes one documented source of run-to-run or machine-to-machine
# variation. They must be set before TensorFlow is first imported.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    CUDA_VISIBLE_DEVICES="" \
    KERAS_BACKEND=tensorflow \
    OMP_NUM_THREADS=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv==0.9.9 \
 && uv export --frozen --no-dev --no-emit-project --format requirements-txt \
      > /tmp/requirements.txt \
 && uv pip install --system --no-cache -r /tmp/requirements.txt

COPY src ./src
RUN pip install --no-cache-dir --no-deps -e .

ENTRYPOINT ["glys-rul"]
CMD ["reproduce"]
