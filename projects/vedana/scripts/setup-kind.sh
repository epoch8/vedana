#!/bin/bash
set -e

CLUSTER_NAME="${KIND_CLUSTER_NAME:-vedana-local}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIND_CONFIG="${SCRIPT_DIR}/kind-config.yaml"

echo "▶ Setting up kind cluster: ${CLUSTER_NAME}"

# Ensure kind exists
command -v kind >/dev/null || {
  echo "kind is not installed"
  exit 1
}

# Delete existing cluster if present
if kind get clusters | grep -q "^${CLUSTER_NAME}$"; then
  echo "Deleting existing cluster..."
  kind delete cluster --name "${CLUSTER_NAME}"
fi

# Create cluster (no special networking needed)
kind create cluster --name "${CLUSTER_NAME}" --config "${KIND_CONFIG}"

# Wait for readiness
kubectl wait --for=condition=Ready nodes --all --timeout=120s \
  --context "kind-${CLUSTER_NAME}"

echo "✔ kind cluster ready"

# Create host kubeconfig
HOST_KUBECONFIG="${HOME}/.kube/config-kind-${CLUSTER_NAME}"
DOCKER_KUBECONFIG="${HOME}/.kube/config-kind-${CLUSTER_NAME}-docker"

mkdir -p "${HOME}/.kube"

# Ensure files, not directories
rm -rf "${HOST_KUBECONFIG}" "${DOCKER_KUBECONFIG}"

kind get kubeconfig --name "${CLUSTER_NAME}" > "${HOST_KUBECONFIG}"

echo "Loading local vedana image into cluster..."
kind load docker-image vedana/vedana:dev --name vedana-local
echo "✔ vedana image loaded"

cp "${HOST_KUBECONFIG}" "${DOCKER_KUBECONFIG}"

sed -i.bak \
  -e "s|https://127.0.0.1:|https://host.docker.internal:|g" \
  -e "s|https://localhost:|https://host.docker.internal:|g" \
  "${DOCKER_KUBECONFIG}"

rm -f "${DOCKER_KUBECONFIG}.bak"


echo "✔ Docker kubeconfig written to ${DOCKER_KUBECONFIG}"

echo ""
echo "NEXT STEPS:"
echo "  export KUBECONFIG=${HOST_KUBECONFIG}"
echo "  kubectl get nodes"
echo ""
echo "Docker containers should mount:"
echo "  ${DOCKER_KUBECONFIG} -> /root/.kube/config"
