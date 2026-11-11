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
# step descriptions (`docker buildx du --verbose`): COPY steps of the `cuda`,
# `rocm` and `ct2-rocm` stages ("[rocm  5/13] COPY ..."), and RUN steps
# ("mount / from exec /bin/sh -c ...") whose command only these Dockerfiles
# run: /build/.venv pip installs of +cu132 / rocm7.14 / rocm7.2 /
# nvidia-cublas-cu12 / the CTranslate2 ROCm wheel, the onnxruntime-gpu swap,
# the ldconfig lists, the CTranslate2 zip download, the GPU variants of
# assert_runtime.py and the GPU stages' final chmod/mkdir step (whose
# command differs from the root Dockerfile's, and which otherwise keeps every
# parent layer referenced). The root Dockerfile's steps never match. Nothing else on
# the daemon is touched.
# ============================================================================
set -eu

# Passed to awk through ENVIRON, not -v: -v processes backslash escapes and
# turns "\[" into "[", which is how the first version matched nothing.
export MATCH='^\[(cuda|rocm|ct2-rocm) +[0-9]+/[0-9]+\]|/build/\.venv/bin/pip .*(\+cu132|rocm7\.14|rocm7\.2|nvidia-cublas-cu12|/tmp/ct2)|ORT_VERSION=.*onnxruntime-gpu|voice-stack-(cuda|rocm)\.conf|ct2-rocm\.zip|accept_coqui_license\.sh && +sed -i|--from=ct2-rocm|assert_runtime\.py --variant (cuda|rocm)|chmod \+x /app/entrypoint\.sh && +mkdir -p /app/voices /app/models'

docker buildx du --verbose > buildx-du.txt
awk '
  /^ID:/          { id = $2 }
  /^Size:/        { size = $2 }
  /^Description:/ { sub(/^Description:[ \t]*/, ""); if ($0 ~ ENVIRON["MATCH"]) print id, size, $0 }
' buildx-du.txt > gpu-cache-records.txt
echo "GPU build cache records: $(wc -l < gpu-cache-records.txt)"
# Fail-closed: a matcher that silently matches nothing looks exactly like a
# clean daemon. The total of what matched is printed so the log shows it.
awk '{ v = $2 + 0; if ($2 ~ /GB$/) t += v; else if ($2 ~ /MB$/) t += v / 1000 } END { printf "matched: %.1f GB\n", t }' gpu-cache-records.txt
cat gpu-cache-records.txt

if [ "${1:-}" != "--apply" ]; then
  echo "dry run; pass --apply to remove them"
  exit 0
fi
docker system df
# Children before parents is not required: prune drops a record once nothing
# references it, so repeat until a pass removes nothing more.
for pass in 1 2 3 4 5; do
  ids=$(docker buildx du --verbose | awk '
    /^ID:/ { id = $2 } /^Description:/ { sub(/^Description:[ \t]*/, ""); if ($0 ~ ENVIRON["MATCH"]) print id }')
  [ -n "$ids" ] || break
  echo "pass ${pass}: $(echo "$ids" | wc -l) records"
  # One id per call: never a filter combination whose AND/OR semantics could
  # widen the prune beyond these records.
  for id in $ids; do docker builder prune -f --filter "id=${id}" >/dev/null || true; done
done
docker system df
