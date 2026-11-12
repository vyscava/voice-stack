#!/bin/sh
# ============================================================================
# Push a GPU image that ci/gpu-image-check.sh has just built and gated.
#
#   sh ci/gpu-image-check.sh cuda && sh ci/gpu-image-publish.sh cuda
#
# Called by the publish:gpu-image:* jobs in .gitlab/ci/gpu-images.yml, in the
# same job and on the same daemon as the check, so the image pushed is the
# exact image that passed the runtime assertions, pip-audit and trivy. Nothing
# is rebuilt here.
#
# Pushes ONE moving tag per variant, ${DOCKER_REGISTRY}/voice-stack:<variant>.
# No per-commit tag: the GPU inputs include src/, so most merges republish,
# and a 9-12 GB image kept per commit would grow the registry by ~20 GB a
# merge. A superseded :cuda / :rocm leaves untagged layers that the registry's
# garbage collection can reclaim.
#
# Needs: GPU_IMAGE (the local tag the check built), DOCKER_REGISTRY,
# DOCKER_REGISTRY_USER, DOCKER_REGISTRY_PASSWORD. Busybox sh compatible.
# ============================================================================
set -eu

VARIANT="${1:?usage: gpu-image-publish.sh cuda|rocm}"
case "$VARIANT" in
  cuda|rocm) ;;
  *) echo "FAIL: unknown variant $VARIANT"; exit 2 ;;
esac
IMAGE="${GPU_IMAGE:?GPU_IMAGE must name the local tag gpu-image-check.sh built}"
REGISTRY="${DOCKER_REGISTRY:?DOCKER_REGISTRY must be set to publish}"
: "${DOCKER_REGISTRY_USER:?DOCKER_REGISTRY_USER must be set to publish}"
: "${DOCKER_REGISTRY_PASSWORD:?DOCKER_REGISTRY_PASSWORD must be set to publish}"
TARGET="${REGISTRY}/voice-stack:${VARIANT}"

# The check must have left the image behind; refuse to push anything else.
docker image inspect "$IMAGE" >/dev/null 2>&1 \
  || { echo "FAIL: ${IMAGE} not found - run ci/gpu-image-check.sh ${VARIANT} first"; exit 1; }

echo "$DOCKER_REGISTRY_PASSWORD" | docker login -u "$DOCKER_REGISTRY_USER" --password-stdin "$REGISTRY"
trap 'docker logout "$REGISTRY" >/dev/null 2>&1 || true' EXIT
docker tag "$IMAGE" "$TARGET"

# The digest comes from the push's own output ("<tag>: digest: sha256:... size:
# N"), which both the classic and the containerd image store print. Local
# RepoDigests are not filled in reliably under the containerd store (the
# docker:29 default), so they are not used as the proof.
docker push "$TARGET" | tee gpu-image-push.log
digest=$(sed -n 's/.*digest: \(sha256:[0-9a-f]\{64\}\).*/\1/p' gpu-image-push.log | tail -n1)
test -n "$digest" || { echo "FAIL: the push of ${TARGET} reported no digest"; exit 1; }
echo "published ${TARGET}@${digest}" | tee gpu-image-published.txt
