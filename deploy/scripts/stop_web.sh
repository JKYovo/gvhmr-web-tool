#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GVHMR_AUTO_FIX=0
source "${SCRIPT_DIR}/environment_helpers.sh"
source "${SCRIPT_DIR}/docker_helpers.sh"

cd "$ROOT_DIR"
ensure_required_commands
ensure_docker_ready
docker_stop_service
