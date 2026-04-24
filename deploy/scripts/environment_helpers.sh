#!/usr/bin/env bash

set -euo pipefail

GVHMR_AUTO_FIX="${GVHMR_AUTO_FIX:-0}"
GVHMR_DOCTOR_QUIET="${GVHMR_DOCTOR_QUIET:-0}"
GVHMR_DOCKER_USE_SUDO="${GVHMR_DOCKER_USE_SUDO:-0}"

doctor_note() {
  if [ "${GVHMR_DOCTOR_QUIET}" != "1" ]; then
    echo "[GVHMR] $*"
  fi
}

doctor_warn() {
  echo "[GVHMR] Warning: $*" >&2
}

doctor_fail() {
  echo "[GVHMR] Error: $*" >&2
  exit 1
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

run_as_root() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  elif command_exists sudo; then
    sudo "$@"
  else
    return 1
  fi
}

apt_install_packages() {
  run_as_root apt-get update
  run_as_root apt-get install -y "$@"
}

is_supported_linux() {
  [ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ]
}

ensure_supported_platform() {
  if ! is_supported_linux; then
    doctor_fail "Supported platform is Linux x86_64 with NVIDIA GPU."
  fi
}

ensure_required_commands() {
  local missing=()
  for cmd in python3; do
    if ! command_exists "$cmd"; then
      missing+=("$cmd")
    fi
  done
  if [ "${#missing[@]}" -ne 0 ]; then
    doctor_fail "Missing required command(s): ${missing[*]}"
  fi
}

ensure_nvidia_driver() {
  if ! command_exists nvidia-smi; then
    doctor_fail "nvidia-smi is missing. Install an NVIDIA driver first."
  fi
  if ! nvidia-smi -L >/dev/null 2>&1; then
    doctor_fail "nvidia-smi is present but no usable NVIDIA GPU was found."
  fi
}

try_install_docker() {
  if ! command_exists apt-get; then
    doctor_fail "Docker is missing. Auto-install currently supports Ubuntu/Debian only."
  fi
  doctor_note "Installing Docker via apt-get..."
  apt_install_packages docker.io
}

enable_docker_service() {
  doctor_note "Starting Docker service..."
  run_as_root systemctl enable --now docker
}

detect_docker_access() {
  if docker info >/dev/null 2>&1; then
    export GVHMR_DOCKER_USE_SUDO=0
    return 0
  fi

  if command_exists sudo && sudo -n docker info >/dev/null 2>&1; then
    export GVHMR_DOCKER_USE_SUDO=1
    return 0
  fi

  return 1
}

ensure_docker_ready() {
  if ! command_exists docker; then
    if [ "${GVHMR_AUTO_FIX}" = "1" ]; then
      try_install_docker
    else
      doctor_fail "Docker is missing. Re-run with --fix on Ubuntu/Debian, or install Docker manually."
    fi
  fi

  if detect_docker_access; then
    return 0
  fi

  if [ "${GVHMR_AUTO_FIX}" = "1" ]; then
    enable_docker_service
    if detect_docker_access; then
      return 0
    fi
    if command_exists sudo && sudo docker info >/dev/null 2>&1; then
      export GVHMR_DOCKER_USE_SUDO=1
      doctor_warn "Using sudo for Docker commands in this shell. Add your user to the docker group to avoid sudo next time."
      return 0
    fi
  fi

  doctor_fail "Cannot access Docker. Start the Docker service or grant this user Docker permission."
}

docker_runtime_has_nvidia() {
  local info
  if [ "${GVHMR_DOCKER_USE_SUDO}" = "1" ]; then
    info="$(sudo docker info --format '{{json .Runtimes}}' 2>/dev/null || true)"
  else
    info="$(docker info --format '{{json .Runtimes}}' 2>/dev/null || true)"
  fi
  grep -q '"nvidia"' <<<"$info"
}

try_install_nvidia_container_toolkit() {
  if ! command_exists apt-get; then
    doctor_fail "NVIDIA Container Toolkit is missing. Auto-install currently supports Ubuntu/Debian only."
  fi

  doctor_note "Installing NVIDIA Container Toolkit via apt-get..."
  if ! apt_install_packages nvidia-container-toolkit; then
    doctor_fail "Failed to install nvidia-container-toolkit automatically. Install it manually, then rerun start_web.sh."
  fi

  if ! command_exists nvidia-ctk; then
    doctor_fail "nvidia-ctk is still unavailable after installing nvidia-container-toolkit."
  fi

  doctor_note "Configuring Docker NVIDIA runtime..."
  run_as_root nvidia-ctk runtime configure --runtime=docker
  run_as_root systemctl restart docker
}

ensure_nvidia_container_runtime() {
  if docker_runtime_has_nvidia; then
    return 0
  fi

  if [ "${GVHMR_AUTO_FIX}" = "1" ]; then
    try_install_nvidia_container_toolkit
    ensure_docker_ready
    if docker_runtime_has_nvidia; then
      return 0
    fi
  fi

  doctor_fail "Docker does not expose the NVIDIA runtime. Install/configure NVIDIA Container Toolkit, or rerun with --fix on Ubuntu/Debian."
}

ensure_runtime_dirs() {
  mkdir -p "${ROOT_DIR}/runtime/checkpoints" \
           "${ROOT_DIR}/runtime/jobs" \
           "${ROOT_DIR}/runtime/batches" \
           "${ROOT_DIR}/runtime/db"
}

run_environment_doctor() {
  doctor_note "Checking host environment..."
  ensure_supported_platform
  ensure_required_commands
  ensure_nvidia_driver
  ensure_docker_ready
  ensure_nvidia_container_runtime
  ensure_runtime_dirs

  if [ "${GVHMR_DOCKER_USE_SUDO}" = "1" ]; then
    doctor_note "Docker access will use sudo in this shell."
  fi
  doctor_note "Environment check passed."
}
