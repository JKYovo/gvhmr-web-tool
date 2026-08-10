#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${ROOT_DIR}/runtime/gvhmr_web.pid"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "GVHMR source Web is not running."
  exit 0
fi

PID="$(cat "${PID_FILE}")"
COMMAND="$(ps -p "${PID}" -o args= 2>/dev/null || true)"
if [[ "${COMMAND}" != *"hmr4d.service.server"* ]]; then
  echo "GVHMR source Web is not running under PID ${PID}."
  rm -f "${PID_FILE}"
  exit 0
fi

kill "${PID}"
for _ in $(seq 1 20); do
  if ! kill -0 "${PID}" 2>/dev/null; then
    rm -f "${PID_FILE}"
    echo "GVHMR source Web stopped."
    exit 0
  fi
  sleep 0.25
done

kill -KILL "${PID}"
rm -f "${PID_FILE}"
echo "GVHMR source Web force-stopped after timeout."
