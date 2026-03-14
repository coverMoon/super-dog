FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV VENV_PATH=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility

WORKDIR /workspace/HIMLoco
SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 \
    python3.8-dev \
    python3.8-venv \
    python3-pip \
    git \
    build-essential \
    ca-certificates \
    libgl1 \
    libegl1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxrandr2 \
    libxi6 \
    libxcursor1 \
    libxinerama1 \
    libvulkan1 \
    mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*

RUN python3.8 -m venv "${VENV_PATH}" && \
    python -m pip install --upgrade pip setuptools wheel

# Match the current training environment: Python 3.8 + torch 1.13.1/cu117.
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu117 \
    numpy==1.23.5 \
    matplotlib==3.7.5 \
    tensorboard==2.14.0 \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    torchaudio==0.13.1+cu117

# Copy only the directories required for training/runtime.
COPY isaacgym ./isaacgym
COPY rsl_rl ./rsl_rl
COPY legged_gym ./legged_gym
COPY README.md ./README.md
COPY LICENSE ./LICENSE

RUN pip install --no-cache-dir -e isaacgym/python && \
    pip install --no-cache-dir -e rsl_rl && \
    pip install --no-cache-dir -e legged_gym

# Build-time smoke test for the pinned runtime stack.
RUN python -c "import sys; print(sys.version)" && \
    python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())" && \
    python -c "import isaacgym; import legged_gym; import rsl_rl; print('imports ok')"

CMD ["/bin/bash"]
