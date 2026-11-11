#!/bin/sh
# ============================================================================
# Remove the BuildKit cache that GPU image builds left on the shared daemon.
#
#   sh ci/gpu-cache-prune.sh          # list what would be removed
#   sh ci/gpu-cache-prune.sh --apply  # remove it
#
# The first GPU builds of !77 ran on the daemon's default builder, whose layer
# cache outlives the job: two pipelines left ~110 GB and filled the runner's
# disk. ci/gpu-image-check.sh now builds in a throwaway builder that is removed
# with its cache, so this only cleans up what came before that (and anything a
# killed job could not remove).
#
# Scoped to records only these Dockerfiles create, identified by their build
# step descriptions: steps of the `cuda`, `rocm` and `ct2-rocm` stages, and
# steps of a stage named exactly `builder` that work on /build/.venv (the root
# Dockerfile's stages are builder-base/-kokoro/-full and never match). Nothing
# else on the daemon is touched.
# ============================================================================
set -eu

MATCH='^\[(cuda|rocm|ct2-rocm) [0-9]+/[0-9]+\]|^\[builder [0-9]+/[0-9]+\] .*/build/\.venv|\+cu132|rocm7\.14|nvidia-cublas-cu12|onnxruntime-gpu|/tmp/ct2|voice-stack-(cuda|rocm)\.conf'

docker buildx du --verbose > buildx-du.txt
awk -v re="$MATCH" '
  /^ID:/          { id = $2 }
  /^Size:/        { size = $2 }
  /^Description:/ { sub(/^Description:[ \t]*/, ""); if ($0 ~ re) print id, size, $0 }
' buildx-du.txt > gpu-cache-records.txt
echo "GPU build cache records: $(wc -l < gpu-cache-records.txt)"
cat gpu-cache-records.txt

if [ "${1:-}" != "--apply" ]; then
  echo "dry run; pass --apply to remove them"
  exit 0
fi
docker system df
# Children before parents is not required: prune drops a record once nothing
# references it, so repeat until a pass removes nothing more.
for pass in 1 2 3 4 5; do
  ids=$(docker buildx du --verbose | awk -v re="$MATCH" '
    /^ID:/ { id = $2 } /^Description:/ { sub(/^Description:[ \t]*/, ""); if ($0 ~ re) print id }')
  [ -n "$ids" ] || break
  echo "pass ${pass}: $(echo "$ids" | wc -l) records"
  # One id per call: never a filter combination whose AND/OR semantics could
  # widen the prune beyond these records.
  for id in $ids; do docker builder prune -f --filter "id=${id}" >/dev/null || true; done
done
docker system df
