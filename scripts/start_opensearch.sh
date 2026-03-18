#!/usr/bin/env bash
# =============================================================================
# start_opensearch.sh
#
# Downloads and runs OpenSearch natively (no Docker required).
# OpenSearch 2.x ships with a bundled JDK — no separate Java install needed.
#
# Usage:
#   bash scripts/start_opensearch.sh            # foreground (Ctrl+C to stop)
#   bash scripts/start_opensearch.sh --daemon   # background, writes PID file
#
# The binary is downloaded once into .opensearch/ and reused on subsequent runs.
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OS_VERSION="${OPENSEARCH_VERSION:-2.13.0}"
ARCH="x64"
INSTALL_DIR="$(pwd)/.opensearch"
DATA_DIR="$(pwd)/data/opensearch"
OS_DIR="${INSTALL_DIR}/opensearch-${OS_VERSION}"
PID_FILE="${INSTALL_DIR}/opensearch.pid"
TARBALL_URL="https://artifacts.opensearch.org/releases/bundle/opensearch/${OS_VERSION}/opensearch-${OS_VERSION}-linux-${ARCH}.tar.gz"
TARBALL_PATH="/tmp/opensearch-${OS_VERSION}-linux-${ARCH}.tar.gz"
DAEMON_MODE=false

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
for arg in "$@"; do
  case $arg in
    --daemon) DAEMON_MODE=true ;;
    *) echo "[warn] Unknown argument: $arg" ;;
  esac
done

# ---------------------------------------------------------------------------
# WSL: vm.max_map_count (OpenSearch requires >= 262144)
# ---------------------------------------------------------------------------
CURRENT_MAP_COUNT=$(sysctl -n vm.max_map_count 2>/dev/null || echo 0)
if [ "${CURRENT_MAP_COUNT}" -lt 262144 ]; then
  echo "[setup] vm.max_map_count=${CURRENT_MAP_COUNT} is too low. Raising to 262144 ..."
  if sudo sysctl -w vm.max_map_count=262144 >/dev/null 2>&1; then
    echo "[setup] vm.max_map_count set successfully."
  else
    echo ""
    echo "  [ERROR] Could not set vm.max_map_count automatically."
    echo "  Run the following manually and retry:"
    echo ""
    echo "    sudo sysctl -w vm.max_map_count=262144"
    echo ""
    echo "  Or add to /etc/sysctl.conf:  vm.max_map_count=262144"
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Download OpenSearch (once)
# ---------------------------------------------------------------------------
if [ ! -d "${OS_DIR}" ]; then
  echo "[setup] OpenSearch ${OS_VERSION} not found. Downloading (~400 MB) ..."
  mkdir -p "${INSTALL_DIR}"
  curl -L --progress-bar -o "${TARBALL_PATH}" "${TARBALL_URL}"
  echo "[setup] Extracting ..."
  tar -xzf "${TARBALL_PATH}" -C "${INSTALL_DIR}"
  rm -f "${TARBALL_PATH}"
  echo "[setup] Installed to ${OS_DIR}"
else
  echo "[setup] Found existing installation: ${OS_DIR}"
fi

# ---------------------------------------------------------------------------
# Write development configuration (overwrites on every start for consistency)
# ---------------------------------------------------------------------------
cat > "${OS_DIR}/config/opensearch.yml" << 'EOF'
cluster.name: benchmark-cluster
node.name: benchmark-node
discovery.type: single-node
plugins.security.disabled: true
network.host: 127.0.0.1
http.port: 9200
transport.port: 9300
EOF

# Lower JVM heap for development (override existing settings)
mkdir -p "${OS_DIR}/config/jvm.options.d"
cat > "${OS_DIR}/config/jvm.options.d/benchmark.options" << 'EOF'
-Xms512m
-Xmx512m
EOF

# ---------------------------------------------------------------------------
# Start OpenSearch
# ---------------------------------------------------------------------------
if [ "${DAEMON_MODE}" = true ]; then
  echo "[start] Starting OpenSearch ${OS_VERSION} in background ..."
  nohup "${OS_DIR}/bin/opensearch" \
    > "${INSTALL_DIR}/opensearch.log" 2>&1 &
  echo $! > "${PID_FILE}"
  echo "[start] PID $(cat "${PID_FILE}") — logs: ${INSTALL_DIR}/opensearch.log"
  echo "[start] Run 'bash scripts/stop_opensearch.sh' to stop."

  # Wait until OpenSearch is ready
  echo -n "[start] Waiting for OpenSearch to become available "
  for i in $(seq 1 30); do
    if curl -s "http://localhost:9200" >/dev/null 2>&1; then
      echo " ready!"
      echo "[start] OpenSearch is up at http://localhost:9200"
      exit 0
    fi
    echo -n "."
    sleep 2
  done
  echo ""
  echo "[warn] OpenSearch did not respond within 60 s. Check logs: ${INSTALL_DIR}/opensearch.log"
  exit 1
else
  echo "[start] Starting OpenSearch ${OS_VERSION} in foreground (Ctrl+C to stop) ..."
  exec "${OS_DIR}/bin/opensearch"
fi
