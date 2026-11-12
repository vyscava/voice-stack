#!/bin/sh
# ============================================================================
# Build one GPU image and gate it, entirely inside one CI job.
#
#   sh ci/gpu-image-check.sh cuda|rocm
#
# Called by .gitlab/ci/gpu-images.yml. The GPU images are proven buildable and
# scanned here but are NEVER pushed: the image only ever exists on the build
# daemon, under a name with no registry in it, and the job's after_script
# removes it. So nothing this script produces can become a deploy candidate.
#
# Steps, each fail-closed:
#   1. docker build (the Dockerfile runs scripts/assert_runtime.py --variant X,
#      so a build that succeeds has already passed the runtime assertions)
#   2. pip freeze + dist-info export of the venv that ships, with the same
#      fake-pass guards as .gitlab/ci/security.yml's .freeze-image
#   3. pip-audit --strict of that export (same pinned pip-audit, no ignores)
#   4. trivy OS-package scan, same policy and guard template as .trivy-image
#
# Why one job and not the build/freeze/audit/trivy split the CPU images use:
# that split hands the image between jobs through the registry. These images
# must not be pushed, and a hand-off through the shared daemon would leave a
# 10+ GB image behind whenever a pipeline is cancelled between jobs.
#
# Needs: GPU_IMAGE (the local tag to build), GPU_BUILDER (the throwaway buildx
# builder's name), CI_JOB_ID, a docker CLI talking to
# the build daemon (DOCKER_HOST). Busybox sh compatible (docker:29 is Alpine).
# ============================================================================
set -eu

VARIANT="${1:?usage: gpu-image-check.sh cuda|rocm}"
case "$VARIANT" in
  cuda) MIN_DISTS=90; TORCH_BUILD='\+cu' ;;
  rocm) MIN_DISTS=90; TORCH_BUILD='\+rocm' ;;
  *) echo "FAIL: unknown variant $VARIANT"; exit 2 ;;
esac
IMAGE="${GPU_IMAGE:?GPU_IMAGE must name the local tag to build}"
JOB="${CI_JOB_ID:-local}"
# Same pins as .gitlab/ci/security.yml: bump them together.
TRIVY_IMAGE="aquasec/trivy@sha256:086971aaf400beebd94e8300fd8ea623774419597169156cec56eec5b00dfb1e"
AUDIT_IMAGE="python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b"
PIP_AUDIT_VERSION="2.10.1"

# ---------------------------------------------------------------- 1. build
# In a THROWAWAY builder, not the daemon's default one. The default builder
# keeps every layer as cache after the job, and these images are 9-12 GB with
# builder stages as large again: the first two pipelines of !77 left ~110 GB on
# the shared runner and filled its disk. A docker-container builder keeps its
# cache in its own volume, which `docker buildx rm` deletes with it (here and
# in the job's after_script), so a build leaves nothing behind but the --load'ed
# image, which after_script removes too. The price is a cold build every time.
BUILDER="${GPU_BUILDER:?GPU_BUILDER must name the throwaway builder}"
BUILDKIT_IMAGE="moby/buildkit@sha256:99368f3e46177c6405b9740e4461d91cee004128daeecca78d4ceac50ec1da5c" # v0.34.1
docker buildx rm -f "$BUILDER" >/dev/null 2>&1 || true
docker buildx create --name "$BUILDER" --driver docker-container \
  --driver-opt "image=${BUILDKIT_IMAGE}" >/dev/null
APT_REFRESH="$(date -u +%F)${APT_REFRESH_FORCE:+-$APT_REFRESH_FORCE}"
start=$(date +%s)
docker buildx build --builder "$BUILDER" --pull --load \
  --file "docker/Dockerfile.${VARIANT}" --tag "$IMAGE" \
  --build-arg APT_REFRESH="$APT_REFRESH" .
seconds=$(( $(date +%s) - start ))
# The image is in the daemon now; the builder and all its cache can go.
docker buildx rm -f "$BUILDER"
bytes=$(docker image inspect -f '{{.Size}}' "$IMAGE")
{
  echo "variant=${VARIANT}"
  echo "build_seconds=${seconds}"
  echo "image_bytes=${bytes}"
  echo "image_mib=$(( bytes / 1024 / 1024 ))"
} | tee gpu-image-metrics.txt

# ---------------------------------------------------------------- 2. freeze
docker run --rm --name "freeze-${VARIANT}-${JOB}" --entrypoint /app/.venv/bin/python "$IMAGE" \
  -m pip freeze --exclude-editable > requirements.freeze.txt
n=$(grep -c '==' requirements.freeze.txt || true)
echo "pinned packages in the ${VARIANT} image: ${n}"
test "$n" -ge "$MIN_DISTS" || { echo "FAIL: only ${n} pinned packages - freeze did not capture the venv"; exit 1; }
# The freeze must show the GPU build that SHIPS, not just a torch.
grep -E "^torch==2\.15\.0${TORCH_BUILD}" requirements.freeze.txt \
  || { echo "FAIL: torch in the ${VARIANT} image is not the ${VARIANT} build"; grep '^torch==' requirements.freeze.txt || true; exit 1; }
if [ "$VARIANT" = "cuda" ]; then
  grep -E '^onnxruntime-gpu==' requirements.freeze.txt || { echo "FAIL: onnxruntime-gpu missing"; exit 1; }
  if grep -E '^onnxruntime==' requirements.freeze.txt; then echo "FAIL: CPU onnxruntime still installed"; exit 1; fi
fi

rm -rf dist-meta && mkdir -p dist-meta
docker run --rm --name "meta-${VARIANT}-${JOB}" --entrypoint sh "$IMAGE" \
  -c 'cd /app/.venv/lib/python3.*/site-packages && tar cf - *.dist-info' | tar xf - -C dist-meta
rm -rf dist-meta/voice_stack-*.dist-info
# Local-version builds (+cu132, +cu130, +rocm7.14, +rocm7.2, +cpu) are not
# PyPI versions. pip-audit --path reads the version from the dist-info
# DIRECTORY NAME (see security.yml), so strip it there and in METADATA.
# Advisories are keyed on the public version.
find dist-meta -name METADATA -exec sed -i -E 's/^Version: ([^+]*)\+.*$/Version: \1/' {} +
for d in dist-meta/*+*.dist-info; do
  [ -e "$d" ] || continue
  mv "$d" "$(echo "$d" | sed -E 's/\+[A-Za-z0-9.]*\.dist-info$/.dist-info/')"
done
if ls -d dist-meta/*+* >/dev/null 2>&1; then echo "FAIL: local versions left in dist-meta"; exit 1; fi
if [ "$VARIANT" = "rocm" ]; then
  # AMD's ROCm runtime packages come from AMD's index (repo.amd.com, via the
  # PyTorch ROCm index). PyPI holds only 0.x placeholders under these names,
  # so pip-audit cannot look 7.14.x up and --strict would fail on "not found",
  # not on an advisory. They are vendor binaries (HIP runtime, rocBLAS,
  # MIOpen...) with no entries in the Python advisory database either way;
  # their OS-level exposure is trivy's job. Excluded BY NAME, and listed.
  echo "excluded from pip-audit (AMD index only, not on PyPI):"
  ls -d dist-meta/rocm-*.dist-info dist-meta/rocm_sdk*.dist-info
  rm -rf dist-meta/rocm-*.dist-info dist-meta/rocm_sdk*.dist-info
fi
n=$(ls -d dist-meta/*.dist-info | wc -l)
echo "dist-info directories to audit: ${n} (minimum ${MIN_DISTS})"
test "$n" -ge "$MIN_DISTS" || { echo "FAIL: dist-info export too small"; exit 1; }
ls -d dist-meta/torch-2.15.0.dist-info

# ---------------------------------------------------------------- 3. pip-audit
# The export goes in over stdin: the daemon cannot see this job's filesystem.
# No ignores (same policy as test:audit:*). The first pass is report-only and
# kept as an artifact; the second is the gate.
audit() {
  tar cf - dist-meta | docker run --rm -i --name "audit-${VARIANT}-${JOB}-$1" "$AUDIT_IMAGE" sh -c "
    set -e; mkdir /w; cd /w; tar xf -
    pip install --no-cache-dir --quiet --disable-pip-version-check pip-audit==${PIP_AUDIT_VERSION} >/dev/null 2>&1
    $2"
}
audit report 'pip-audit --progress-spinner off --path dist-meta -f json 2>/dev/null || true' > pip-audit-all.json
audit gate 'pip-audit --strict --progress-spinner off --path dist-meta'
echo "audited ${n} distributions"

# ---------------------------------------------------------------- 4. trivy
# The trivy container talks to the same daemon through its socket (the daemon
# resolves the bind mount on its own filesystem) and reads the image from it,
# as .trivy-image does over DOCKER_HOST. The guard template is copied in, the
# report copied out. --exit-code 1 gates on findings; the guard then fails a
# scan that analysed nothing (trivy exits 0 with an empty report on a DB or
# egress failure).
TRIVY_CTR="trivy-${VARIANT}-${JOB}"
docker rm -f "$TRIVY_CTR" >/dev/null 2>&1 || true
docker create --name "$TRIVY_CTR" --entrypoint sh \
  -v /var/run/docker.sock:/var/run/docker.sock -e TRIVY_NO_PROGRESS=true \
  "$TRIVY_IMAGE" -c '
    trivy --version
    trivy image "$0" --docker-host unix:///var/run/docker.sock --image-src docker \
      --scanners vuln --pkg-types os --list-all-pkgs \
      --severity HIGH,CRITICAL --ignore-unfixed \
      --format json --output /tmp/trivy-report.json --exit-code 1
    rc=$?
    trivy convert --format table --severity HIGH,CRITICAL /tmp/trivy-report.json || true
    trivy convert --format template --template @/tmp/trivy-guard.tpl /tmp/trivy-report.json > /tmp/counts.txt || rc=97
    exit $rc' "$IMAGE" >/dev/null
docker cp ci/trivy-guard.tpl "$TRIVY_CTR:/tmp/trivy-guard.tpl"
set +e
docker start -a "$TRIVY_CTR"
trivy_rc=$?
set -e
docker cp "$TRIVY_CTR:/tmp/trivy-report.json" trivy-report.json || true
docker cp "$TRIVY_CTR:/tmp/counts.txt" trivy-counts.txt || true
docker rm -f "$TRIVY_CTR" >/dev/null 2>&1 || true
if [ ! -s trivy-report.json ] || [ ! -s trivy-counts.txt ]; then
  echo "FAIL: trivy report missing, empty or unparsable - scan did not run (rc=${trivy_rc})"
  exit 1
fi
awk '{ pkgs += $1; vulns += $2 }
  END {
    if (pkgs < 1) { print "FAIL: no analyzed packages in any Result - image was not scanned"; exit 1 }
    print "ok   trivy analyzed " pkgs " packages; " vulns " vulnerability entries"
  }' trivy-counts.txt
if [ "$trivy_rc" -ne 0 ]; then
  echo "FAIL: trivy found fixable HIGH/CRITICAL OS vulnerabilities (rc=${trivy_rc}); table above"
  exit 1
fi

echo "OK: ${VARIANT} image built, runtime-asserted, audited and scanned (${seconds}s, $(( bytes / 1024 / 1024 )) MiB)"
