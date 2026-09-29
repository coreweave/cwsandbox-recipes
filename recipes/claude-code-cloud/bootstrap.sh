#!/bin/bash
set -eu
umask 077
mkdir -p /workspace
exec > >(tee -a /workspace/bootstrap.log) 2>&1
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq curl ca-certificates git libatomic1
curl -fsSL https://claude.ai/install.sh -o /tmp/claude-install.sh
bash /tmp/claude-install.sh 2.1.284
export PATH="/root/.local/bin:$PATH"
export DISABLE_AUTOUPDATER=1
if [ "$1" = orchestrator ]; then
    python -m pip install --disable-pip-version-check --root-user-action=ignore -r /opt/recipe/requirements.txt
    exec python /opt/recipe/cloud.py supervise
fi
exec claude self-hosted-runner \
    --base-dir /workspace/sessions --capacity 1 \
    --use-anthropic-git-proxy --configure-git \
    --drain-grace-sec 0 --exit-if-unused-min 5 \
    --release-idle-session-min "$WORKER_IDLE_MINUTES" \
    --retire-at "$WORKER_RETIRE_AT" \
    --client-label "$WORKER_LABEL" --log-file /workspace/runner.log
