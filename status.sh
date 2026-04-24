#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${GVHMR_PORT:-7860}"
source "${ROOT_DIR}/tools/app/docker_helpers.sh"

cd "$ROOT_DIR"
docker_service_status
echo
echo "Expected Web URL: http://127.0.0.1:${PORT}/ui"
