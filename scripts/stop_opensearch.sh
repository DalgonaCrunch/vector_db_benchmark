#!/usr/bin/env bash
# =============================================================================
# stop_opensearch.sh  –  Stop the OpenSearch process started by start_opensearch.sh
# =============================================================================
set -euo pipefail

PID_FILE="$(pwd)/.opensearch/opensearch.pid"

if [ -f "${PID_FILE}" ]; then
  PID=$(cat "${PID_FILE}")
  if kill -0 "${PID}" 2>/dev/null; then
    echo "[stop] Stopping OpenSearch (PID ${PID}) ..."
    kill "${PID}"
    rm -f "${PID_FILE}"
    echo "[stop] Done."
  else
    echo "[stop] Process ${PID} is not running. Removing stale PID file."
    rm -f "${PID_FILE}"
  fi
else
  echo "[stop] No PID file found at ${PID_FILE}."
  echo "       Trying to locate OpenSearch by process name ..."
  if pkill -f "opensearch" 2>/dev/null; then
    echo "[stop] OpenSearch processes terminated."
  else
    echo "[stop] No OpenSearch processes found."
  fi
fi
