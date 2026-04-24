#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PORT="${GVHMR_PORT:-7860}"
source "${SCRIPT_DIR}/docker_helpers.sh"

cd "$ROOT_DIR"
docker_service_status
echo
echo "Expected Web URL: http://127.0.0.1:${PORT}/ui"
