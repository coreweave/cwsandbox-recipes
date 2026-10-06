#!/usr/bin/env bash
set -euo pipefail
scripts_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python "$scripts_dir/install_nemo_source.py"
echo "Installing pinned Python dependencies"
python -m pip install --timeout 30 --retries 3 -r "$scripts_dir/requirements-smoke.txt"
