#!/bin/bash
set -euo pipefail

OLLAMA_HOST="${OLLAMA_BASE_URL:-http://ollama:11434}"
OLLAMA_HOST="${OLLAMA_HOST%/}"
INDEX_PATH="${FAISS_INDEX_DIR:-./storage/faiss_index}/index.faiss"
MODEL="${MODEL_NAME:-qwen2.5:7b}"

wait_for_ollama() {
  if [ "${MODEL_BACKEND:-ollama}" != "ollama" ]; then
    return 0
  fi

  echo "Waiting for Ollama at ${OLLAMA_HOST}..."
  for _ in $(seq 1 90); do
    if curl -sf "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
      echo "Ollama is ready."
      return 0
    fi
    sleep 2
  done

  echo "Timed out waiting for Ollama at ${OLLAMA_HOST}" >&2
  exit 1
}

pull_model_if_needed() {
  if [ "${MODEL_BACKEND:-ollama}" != "ollama" ]; then
    return 0
  fi
  if [ "${OLLAMA_PULL_MODEL:-true}" != "true" ]; then
    return 0
  fi

  if curl -sf "${OLLAMA_HOST}/api/tags" | grep -Fq "\"${MODEL}\""; then
    echo "Model ${MODEL} is already available."
    return 0
  fi

  echo "Pulling Ollama model ${MODEL} (this may take several minutes)..."
  curl -sf "${OLLAMA_HOST}/api/pull" -H "Content-Type: application/json" -d "{\"name\":\"${MODEL}\"}"
  echo
}

build_index_if_needed() {
  if [ -f "${INDEX_PATH}" ]; then
    echo "FAISS index found at ${INDEX_PATH}"
    return 0
  fi

  echo "Building FAISS index..."
  python scripts/build_index.py
}

case "${1:-serve}" in
  serve)
    wait_for_ollama
    pull_model_if_needed
    build_index_if_needed
    exec python -m src.api
    ;;
  build-index)
    build_index_if_needed
    ;;
  wait-ollama)
    wait_for_ollama
    ;;
  pull-model)
    wait_for_ollama
    pull_model_if_needed
    ;;
  bash)
    exec bash
    ;;
  *)
    exec "$@"
    ;;
esac
