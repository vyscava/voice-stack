#!/bin/sh
# ============================================================================
# Remove the BuildKit cache that GPU image builds left on the shared daemon.
#
#   sh ci/gpu-cache-prune.sh          # list what would be removed
#   sh ci/gpu-cache-prune.sh --apply  # remove it
#
# The first GPU builds ran on the daemon's default builder, whose layer
# cache outlives the job: two pipelines left ~110 GB and filled the runner's
# disk. ci/gpu-image-check.sh now builds in a throwaway builder that is removed
# with its cache, so this only cleans up what came before that (and anything a
# killed job could not remove).
#
# Scoped to records only these Dockerfiles create, identified by their build
# step descriptions (`docker buildx du --verbose`): COPY steps of the
# `voice-stack-cuda`, `voice-stack-rocm` and `voice-stack-ct2-rocm` stages
# ("[voice-stack-rocm  5/13] COPY ..."). The stages carry the project prefix
# because the daemon is shared: a bare `cuda` or `rocm` stage name would also
# match other projects' GPU builds. Also RUN steps
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
export MATCH='^\[voice-stack-(cuda|rocm|ct2-rocm) +[0-9]+/[0-9]+\]|/build/\.venv/bin/pip .*(\+cu132|rocm7\.14|rocm7\.2|nvidia-cublas-cu12|/tmp/ct2)|ORT_VERSION=.*onnxruntime-gpu|voice-stack-(cuda|rocm)\.conf|ct2-rocm\.zip|accept_coqui_license\.sh && +sed -i|--from=voice-stack-ct2-rocm|assert_runtime\.py --variant (cuda|rocm)|chmod \+x /app/entrypoint\.sh && +mkdir -p /app/voices /app/models'

# The full listing holds every project's build-step command lines on this
# shared daemon (possibly with inline secrets), so it stays in a temp file and
# is never written to the job directory, logged or kept as an artifact. Only
# the matched records (this project's own steps) are.
DU=$(mktemp)
trap 'rm -f "$DU"' EXIT
docker buildx du --verbose > "$DU"
awk '
  /^ID:/          { id = $2 }
  /^Size:/        { size = $2 }
  /^Description:/ { sub(/^Description:[ \t]*/, ""); if ($0 ~ ENVIRON["MATCH"]) print id, size, $0 }
' "$DU" > gpu-cache-records.txt
echo "GPU build cache records: $(wc -l < gpu-cache-records.txt)"
# Fail-closed: a matcher that silently matches nothing looks exactly like a
# clean daemon. The total of what matched is printed so the log shows it.
awk '{ v = $2 + 0; if ($2 ~ /GB$/) t += v; else if ($2 ~ /MB$/) t += v / 1000 } END { printf "matched: %.1f GB\n", t }' gpu-cache-records.txt
cat gpu-cache-records.txt

# Throwaway builders a GPU build job left behind. ci/gpu-image-check.sh names
# them gpu-<variant>-<CI_JOB_ID>, and buildx runs each as a container named
# buildx_buildkit_<builder>0 with a <container>_state volume holding its cache.
# The job's after_script removes them, but a CANCELLED job can skip it (seen
# 2026-11-11: job 89334's builder was still up 5 hours later). This job shares
# resource_group gpu-image-build with the builds, so no GPU build is running
# while it does and every such builder is an orphan.
ORPHAN_RE='^buildx_buildkit_gpu-(cuda|rocm)-[0-9]+$'
orphans=$(docker ps -a --format '{{.Names}}' | grep -E "$ORPHAN_RE" || true)
echo "orphan GPU builders: $(printf '%s' "$orphans" | grep -c . || true)"
[ -z "$orphans" ] || printf '%s\n' "$orphans"

if [ "${1:-}" != "--apply" ]; then
  echo "dry run; pass --apply to remove them"
  exit 0
fi
for c in $orphans; do
  docker rm -f "$c" >/dev/null || true
  docker volume rm "${c}_state" >/dev/null 2>&1 || true
  echo "removed builder ${c} and volume ${c}_state"
done
docker system df
# A record referenced by a child survives its prune, so this walks the chains
# one generation per pass, until a pass removes nothing more (or 20 passes).
prev=-1
for pass in $(seq 1 20); do
  ids=$(docker buildx du --verbose | awk '
    /^ID:/ { id = $2 } /^Description:/ { sub(/^Description:[ \t]*/, ""); if ($0 ~ ENVIRON["MATCH"]) print id }')
  [ -n "$ids" ] || break
  count=$(echo "$ids" | wc -l)
  [ "$count" -ne "$prev" ] || { echo "pass ${pass}: ${count} records left, none removable"; break; }
  prev=$count
  echo "pass ${pass}: ${count} records"
  # One id per call: never a filter combination whose AND/OR semantics could
  # widen the prune beyond these records.
  for id in $ids; do docker builder prune -f --filter "id=${id}" >/dev/null || true; done
done
docker system df
