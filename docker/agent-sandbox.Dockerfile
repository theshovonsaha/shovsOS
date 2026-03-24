# Dedicated execution image for agent command sandboxing.
# Tuned for interactive coding tasks with Python + Node + common CLI tools.

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    jq \
    less \
    nano \
    vim-tiny \
    procps \
    iproute2 \
    iputils-ping \
    netcat-openbsd \
    build-essential \
    make \
    ripgrep \
    fd-find \
    tree \
    nodejs \
    npm \
    tini \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir \
    pytest \
    requests \
    httpx \
    rich \
    pydantic

RUN useradd --create-home --uid 1000 sandbox
WORKDIR /workspace

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["sleep", "infinity"]
