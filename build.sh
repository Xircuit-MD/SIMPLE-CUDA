#!/usr/bin/env bash
# Build and push the CUDA vector-add image to Docker Hub.
#
# Usage:
#   DOCKERHUB_USER=youruser ./build.sh           # build + push :latest
#   DOCKERHUB_USER=youruser TAG=v1 ./build.sh    # custom tag
#   DOCKERHUB_USER=youruser PUSH=0 ./build.sh    # build only
#
# RunPod serverless GPUs are linux/amd64, so we force that platform.

set -euo pipefail

: "${DOCKERHUB_USER:?Set DOCKERHUB_USER to your Docker Hub username}"
TAG="${TAG:-latest}"
PUSH="${PUSH:-1}"
IMAGE="${DOCKERHUB_USER}/cuda-vecadd:${TAG}"

echo ">> Building ${IMAGE} (linux/amd64)"
docker build --platform linux/amd64 -t "${IMAGE}" .

if [[ "${PUSH}" == "1" ]]; then
    echo ">> Pushing ${IMAGE}"
    docker push "${IMAGE}"
fi

echo ">> Done: ${IMAGE}"
