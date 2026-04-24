#!/usr/bin/env bash

GVHMR_IMAGE_NAME="${GVHMR_IMAGE_NAME:-gvhmr-web:latest}"
GVHMR_CONTAINER_NAME="${GVHMR_CONTAINER_NAME:-gvhmr-web}"

docker_has_compose_plugin() {
  docker compose version >/dev/null 2>&1
}

docker_has_legacy_compose() {
  command -v docker-compose >/dev/null 2>&1
}

docker_use_compose() {
  docker_has_compose_plugin || docker_has_legacy_compose
}

docker_compose_cmd() {
  if docker_has_compose_plugin; then
    docker compose "$@"
  else
    docker-compose "$@"
  fi
}

docker_container_exists() {
  docker ps -a --format '{{.Names}}' | grep -Fxq "$GVHMR_CONTAINER_NAME"
}

docker_container_running() {
  docker ps --format '{{.Names}}' | grep -Fxq "$GVHMR_CONTAINER_NAME"
}

docker_build_runtime() {
  if [ "${GVHMR_SKIP_BUILD:-0}" = "1" ]; then
    docker image inspect "$GVHMR_IMAGE_NAME" >/dev/null 2>&1 || {
      echo "Image $GVHMR_IMAGE_NAME not found; cannot skip build." >&2
      return 1
    }
    echo "Skipping Docker build; reusing image $GVHMR_IMAGE_NAME"
    return 0
  fi
  if docker_use_compose; then
    docker_compose_cmd build gvhmr-web
  else
    docker build -t "$GVHMR_IMAGE_NAME" -f Dockerfile .
  fi
}

docker_run_oneoff() {
  if docker_use_compose; then
    docker_compose_cmd run --rm --no-deps gvhmr-web "$@"
  else
    docker run --rm \
      --gpus all \
      --shm-size 8g \
      -e GVHMR_HOST=0.0.0.0 \
      -e GVHMR_PORT="${GVHMR_PORT}" \
      -e GVHMR_CHECKPOINT_ROOT=/app/runtime/checkpoints \
      -e GVHMR_OUTPUT_ROOT=/app/runtime/jobs \
      -e GVHMR_BATCH_ROOT=/app/runtime/batches \
      -e GVHMR_DB_PATH=/app/runtime/db/job_db.sqlite \
      -v "${ROOT_DIR}/runtime/checkpoints:/app/runtime/checkpoints" \
      -v "${ROOT_DIR}/runtime/jobs:/app/runtime/jobs" \
      -v "${ROOT_DIR}/runtime/batches:/app/runtime/batches" \
      -v "${ROOT_DIR}/runtime/db:/app/runtime/db" \
      "$GVHMR_IMAGE_NAME" "$@"
  fi
}

docker_start_service() {
  if docker_use_compose; then
    docker_compose_cmd up -d gvhmr-web
  else
    if docker_container_exists; then
      docker rm -f "$GVHMR_CONTAINER_NAME" >/dev/null
    fi
    docker run -d \
      --name "$GVHMR_CONTAINER_NAME" \
      --gpus all \
      --shm-size 8g \
      -p "${GVHMR_PORT_BIND}" \
      -e GVHMR_HOST=0.0.0.0 \
      -e GVHMR_PORT="${GVHMR_PORT}" \
      -e GVHMR_CHECKPOINT_ROOT=/app/runtime/checkpoints \
      -e GVHMR_OUTPUT_ROOT=/app/runtime/jobs \
      -e GVHMR_BATCH_ROOT=/app/runtime/batches \
      -e GVHMR_DB_PATH=/app/runtime/db/job_db.sqlite \
      -v "${ROOT_DIR}/runtime/checkpoints:/app/runtime/checkpoints" \
      -v "${ROOT_DIR}/runtime/jobs:/app/runtime/jobs" \
      -v "${ROOT_DIR}/runtime/batches:/app/runtime/batches" \
      -v "${ROOT_DIR}/runtime/db:/app/runtime/db" \
      "$GVHMR_IMAGE_NAME" \
      python3 -m hmr4d.service.server --host 0.0.0.0 --port "${GVHMR_PORT}"
  fi
}

docker_stop_service() {
  if docker_use_compose; then
    docker_compose_cmd down
  else
    if docker_container_exists; then
      docker rm -f "$GVHMR_CONTAINER_NAME"
    else
      echo "GVHMR Web container is not running."
    fi
  fi
}

docker_service_status() {
  if docker_use_compose; then
    docker_compose_cmd ps
  else
    docker ps -a --filter "name=^/${GVHMR_CONTAINER_NAME}$"
  fi
}
