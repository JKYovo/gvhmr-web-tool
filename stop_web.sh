#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${ROOT_DIR}/tools/app/docker_helpers.sh"

cd "$ROOT_DIR"
docker_stop_service
