#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BIN="${CONDA_EXE:-}"
ENV_NAME="${GVHMR_CONDA_ENV:-gvhmr}"
HOST="${GVHMR_HOST:-127.0.0.1}"
PID_FILE="${ROOT_DIR}/runtime/gvhmr_web.pid"
LOG_FILE="${ROOT_DIR}/runtime/gvhmr_web.log"
DEFAULT_CORE_ROOT="${ROOT_DIR}/../gvhmr-core-opt"
CORE_ROOT="${GVHMR_CORE_ROOT:-}"

if [[ -z "${CONDA_BIN}" ]]; then
  for candidate in "${HOME}/miniforge3/bin/conda" "${HOME}/miniconda3/bin/conda"; do
    if [[ -x "${candidate}" ]]; then
      CONDA_BIN="${candidate}"
      break
    fi
  done
fi
if [[ -z "${CONDA_BIN}" ]]; then
  echo "Cannot find conda. Set CONDA_EXE first." >&2
  exit 1
fi

PYTHON_BIN="$(${CONDA_BIN} run -n "${ENV_NAME}" python -c 'import sys; print(sys.executable)')"
PYTHON_BIN="$(printf '%s' "${PYTHON_BIN}" | tail -1)"
if [[ -z "${CORE_ROOT}" && -f "${DEFAULT_CORE_ROOT}/hmr4d/__init__.py" ]]; then
  CORE_ROOT="${DEFAULT_CORE_ROOT}"
fi

port_is_free() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket() as sock:
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", port)) != 0 else 1)
PY
}

if [[ -n "${GVHMR_PORT:-}" ]]; then
  PORT="${GVHMR_PORT}"
  if ! port_is_free "${PORT}"; then
    echo "Port ${PORT} is already in use." >&2
    exit 1
  fi
else
  PORT=7860
  while ! port_is_free "${PORT}"; do
    PORT=$((PORT + 1))
    if (( PORT > 7899 )); then
      echo "No free port found between 7860 and 7899." >&2
      exit 1
    fi
  done
fi

mkdir -p "${ROOT_DIR}/runtime"
if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "GVHMR Web is already running with PID $(cat "${PID_FILE}")."
  exit 0
fi

ENV_ARGS=(
  "PYTHONUNBUFFERED=1"
  "PYTHONNOUSERSITE=1"
  "GVHMR_HOST=${HOST}"
  "GVHMR_PORT=${PORT}"
  "GVHMR_CHECKPOINT_ROOT=${GVHMR_CHECKPOINT_ROOT:-${ROOT_DIR}/inputs/checkpoints}"
  "GVHMR_CORE_PYTHON=${GVHMR_CORE_PYTHON:-${PYTHON_BIN}}"
)
if [[ -n "${CORE_ROOT}" ]]; then
  ENV_ARGS+=("GVHMR_CORE_ROOT=${CORE_ROOT}")
fi

cd "${ROOT_DIR}"
setsid env "${ENV_ARGS[@]}" "${PYTHON_BIN}" -m hmr4d.service.server --host "${HOST}" --port "${PORT}" >"${LOG_FILE}" 2>&1 &
PID="$!"
printf '%s\n' "${PID}" > "${PID_FILE}"

for _ in $(seq 1 60); do
  if curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "GVHMR Web is ready."
    echo "PID: ${PID}"
    echo "URL: http://127.0.0.1:${PORT}/"
    echo "Core: ${CORE_ROOT:-embedded}"
    echo "Log: ${LOG_FILE}"
    exit 0
  fi
  if ! kill -0 "${PID}" 2>/dev/null; then
    echo "GVHMR Web failed to start. Recent log:" >&2
    tail -100 "${LOG_FILE}" >&2 || true
    exit 1
  fi
  sleep 1
done

echo "GVHMR Web is still starting. Check ${LOG_FILE}." >&2
exit 1
