FROM ghcr.io/tailscale/tailcat:v0.7.0@sha256:cf2ed152ee7092dafb95f363dafdebfb1d784ac634a5c4aa01ee408817bbec8c AS tailcat
FROM python:3.12-slim-bookworm

RUN --mount=type=tmpfs,target=/var/lib/apt/lists \
    --mount=type=tmpfs,target=/var/cache/apt \
    --mount=type=tmpfs,target=/tmp \
    apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl socat
COPY --from=tailcat /usr/local/bin/tailcat /usr/local/bin/tailcat
COPY sandbox_service.py /opt/recipes/sandbox_service.py
ENV HOME=/tmp
USER 10001:10001
WORKDIR /tmp
CMD ["sleep", "1800"]
