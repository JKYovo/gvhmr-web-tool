#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GVHMR_AUTO_FIX=0

usage() {
  cat <<'EOF'
Usage: bash doctor.sh [--fix]

Checks:
  - Linux x86_64
  - python3
  - NVIDIA driver / nvidia-smi
  - Docker installation and daemon access
  - NVIDIA Container Toolkit runtime exposure
  - runtime/ directory skeleton

Options:
  --fix   Attempt automatic repair on Ubuntu/Debian for:
          - Docker installation
          - docker.service startup
          - NVIDIA Container Toolkit install/configure
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --fix)
      GVHMR_AUTO_FIX=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

export GVHMR_AUTO_FIX
source "${SCRIPT_DIR}/environment_helpers.sh"

cd "$ROOT_DIR"
run_environment_doctor
