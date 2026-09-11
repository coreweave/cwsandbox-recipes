#!/usr/bin/env bash
# Build locally and test before optionally publishing a CPU-only tau-bench image.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || "$1" == -* || ( $# -eq 2 && "$2" != "--push" ) ]]; then
    echo "Usage: bash scripts/build_env_image.sh IMAGE[:TAG] [--push]" >&2
    exit 2
fi
image_ref="$1"
recipe_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Deliberately use a dedicated context, never the recipe root (which has .env).
docker buildx build --platform linux/amd64 --load --tag "$image_ref" \
    "$recipe_root/docker/taubench-env"

# Run the actual uploaded server's imports offline and verify its writable path.
docker run --rm --platform linux/amd64 --network none \
    --mount "type=bind,src=$recipe_root/verl_taubench/sandbox/env_server.py,dst=/tmp/env_server.py,readonly" \
    --entrypoint python "$image_ref" -c '
import os, runpy
from pathlib import Path
runpy.run_path("/tmp/env_server.py", run_name="image_smoke_test")
assert os.getuid() != 0, "runtime must be non-root"
Path("/srv/write-test").write_text("ok")
assert not Path("/srv/.env").exists()
print("tau-bench image smoke test passed")
'

if [[ "${2:-}" == "--push" ]]; then
    docker push "$image_ref"
    echo "Use the digest from the push output for TAUBENCH_ENV_IMAGE."
else
    echo "Image tested locally; rerun with --push to publish it."
fi
