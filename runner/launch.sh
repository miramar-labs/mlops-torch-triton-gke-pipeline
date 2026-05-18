#!/bin/bash
set -euo pipefail

TOKEN="${1:?Usage: launch.sh <RUNNER_TOKEN> [REPO_URL]}"
REPO_URL="${2:-https://github.com/miramar-labs-org/mlops-torch-triton-gke-pipeline}"

# Derive a unique container name from the repo slug
REPO_SLUG="${REPO_URL##*/}"
CONTAINER_NAME="github-runner-${REPO_SLUG}"

# Detect arch
case "$(uname -m)" in
  x86_64)  RUNNER_NAME=MSIWSL2;  RUNNER_LABELS=msi-wsl2  ;;
  aarch64) RUNNER_NAME=DGXSPARK; RUNNER_LABELS=dgx-spark ;;
  *) echo "Unknown arch: $(uname -m)"; exit 1 ;;
esac

IMAGE=ghcr.io/miramar-labs-org/github-runner-mlops-torch-triton-gke-pipeline:latest
DOCKER_GID=$(stat -c '%g' /var/run/docker.sock)

echo "Runner:    $RUNNER_NAME ($RUNNER_LABELS) on $(uname -m)"
echo "Container: $CONTAINER_NAME"
echo "Image:     $IMAGE"

# Stop and remove existing container for this repo if running
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Stopping existing ${CONTAINER_NAME} container..."
  docker rm -f "$CONTAINER_NAME"
fi

docker pull "$IMAGE"

docker run -d --restart unless-stopped \
  --name "$CONTAINER_NAME" \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --group-add "$DOCKER_GID" \
  -e REPO_URL="$REPO_URL" \
  -e RUNNER_TOKEN="$TOKEN" \
  -e RUNNER_NAME="$RUNNER_NAME" \
  -e RUNNER_LABELS="$RUNNER_LABELS" \
  "$IMAGE"

echo "Started. Logs: docker logs -f $CONTAINER_NAME"
