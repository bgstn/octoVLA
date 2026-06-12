FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /octo

# Copy project files (build context should be the repo root)
COPY . .

RUN pip install --upgrade pip

# Install the package in editable mode
RUN pip install -e .

# Install requirements excluding the three packages that conflict on aarch64:
#   - tensorflow_text: no aarch64 wheel exists for TF <=2.17 era
#   - tensorflow_graphics: requires tensorflow-addons which has no aarch64 wheel
#   - dlimp: pins tensorflow==2.15.0 strictly, which conflicts with tensorflow-text resolution
RUN grep -vE "tensorflow_text|tensorflow_graphics|dlimp" requirements.txt \
    > /tmp/req_base.txt && pip install -r /tmp/req_base.txt

# tensorflow-text: earliest aarch64 wheel is 2.18.x (needs TF>=2.18), but we're on TF 2.15.
# Install with --no-deps to bypass the TF version check at install time.
RUN pip install --no-deps "tensorflow-text>=2.13.0"

# tensorflow-graphics: tensorflow-addons has no aarch64 wheel; skip that transitive dep.
RUN pip install --no-deps tensorflow_graphics==2021.12.3

# dlimp: install without deps so pip doesn't try to re-resolve tensorflow==2.15.0
# against the tensorflow-text we just installed.
RUN pip install --no-deps \
    "dlimp @ git+https://github.com/kvablack/dlimp@5edaa4691567873d495633f2708982b42edf1972"

# JAX GPU: cuda12_pip provides aarch64 CUDA wheels; cuda11_pip does not.
RUN pip install --upgrade "jax[cuda12_pip]==0.4.20" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

CMD ["/bin/bash"]
