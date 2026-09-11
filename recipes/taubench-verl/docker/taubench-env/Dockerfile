# syntax=docker/dockerfile:1
# No recipe files or credentials enter this public image's build context.
ARG PYTHON_IMAGE=python:3.11-slim-bookworm@sha256:528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84
FROM ${PYTHON_IMAGE} AS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
RUN /opt/venv/bin/pip install --no-cache-dir \
    'git+https://github.com/sierra-research/tau-bench.git@59a200c6d575d595120f1cb70fea53cef0632f6b'

FROM ${PYTHON_IMAGE} AS runtime
COPY --from=dependencies /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN useradd --create-home --uid 10001 sandbox \
    && mkdir -p /srv && chown sandbox:sandbox /srv
USER sandbox
WORKDIR /srv
# The backend uploads env_server.py and starts it with a per-sandbox token.
CMD ["sleep", "infinity"]
