# Iteration-2 development image: Python plus the Phase 0-1 toolchain. No models; GPU work stays in canary
# (and later a dedicated serving container). Build context is the repository root.
FROM python:3.12-slim

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        postgresql-client \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

ENV TZ=America/New_York \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        "psycopg[binary]>=3.2" \
        pgvector \
        pytest \
        pytest-asyncio \
        ruff \
        httpx

# Machine-wide git policy is mounted from the host at /root/.git-hooks and /root/.gitignore_global.
RUN git config --global core.hooksPath /root/.git-hooks \
    && git config --global core.excludesFile /root/.gitignore_global \
    && git config --global --add safe.directory /workspace

WORKDIR /workspace
CMD ["tail", "-f", "/dev/null"]
