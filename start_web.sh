#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${GVHMR_PORT:-7860}"
export GVHMR_PORT="$PORT"
export GVHMR_PORT_BIND="127.0.0.1:${PORT}:${PORT}"
source "${ROOT_DIR}/tools/app/docker_helpers.sh"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

check_port_free() {
  python3 - "$PORT" <<'PY'
import socket, sys
port = int(sys.argv[1])
with socket.socket() as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if sock.connect_ex(("127.0.0.1", port)) == 0:
        raise SystemExit(f"Port {port} is already in use.")
PY
}

gpu_runtime_check() {
  docker_run_oneoff \
    python3 -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 'CUDA is unavailable inside the container.')"
}

wait_for_health() {
  python3 - "$PORT" <<'PY'
import json
import sys
import time
import urllib.request

port = int(sys.argv[1])
url = f"http://127.0.0.1:{port}/health"
deadline = time.time() + 300
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            if response.status == 200:
                payload = json.loads(response.read().decode("utf-8"))
                print(payload["status"])
                raise SystemExit(0)
    except Exception:
        time.sleep(1)
raise SystemExit("Timed out waiting for GVHMR Web to become healthy.")
PY
}

open_browser() {
  local url="$1"
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$url" >/dev/null 2>&1 || true
  fi
}

cd "$ROOT_DIR"

require_cmd docker
require_cmd python3

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required." >&2
  exit 1
fi

docker info >/dev/null
if ! docker_container_running; then
  check_port_free
fi

mkdir -p runtime/checkpoints runtime/jobs runtime/batches runtime/db

docker_build_runtime
gpu_runtime_check
docker_run_oneoff python3 -m hmr4d.service.assets --checkpoint-root /app/runtime/checkpoints
docker_start_service
wait_for_health >/dev/null

URL="http://127.0.0.1:${PORT}/ui"
echo "GVHMR Web is ready at ${URL}"
open_browser "$URL"
