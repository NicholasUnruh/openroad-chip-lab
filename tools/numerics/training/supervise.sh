#!/usr/bin/env bash
# Runs every training run listed in $RUNS/queue.txt to completion, one chunk at a time, on a
# GPU that is shared with interactive use. Each queue line is "<name> <train.py arguments>".
#
# - Singleton: a second copy exits at once, so a periodic launcher can start it blindly.
# - Each chunk (train.py, at most --chunk-minutes) holds $GPU_LOCK; between chunks the lock is
#   released for a short gap so other GPU jobs waiting on it can run.
# - train.py exits 75 when its chunk ends or when another process appears on the GPU (it saves a
#   checkpoint first); after a yield the supervisor waits before retrying.
# - Bounded: at most $MAX_CHUNKS chunks per invocation and $MAX_FAILURES failed chunks per run.
set -u
cd "$(dirname "$0")"
DATA=${DATA:?set DATA to the token directory}
RUNS=${RUNS:?set RUNS to the run directory}
GPU_LOCK=${GPU_LOCK:-$HOME/.cache/gpu.lock}
MAX_CHUNKS=${MAX_CHUNKS:-1000}
MAX_FAILURES=${MAX_FAILURES:-5}
mkdir -p "$RUNS" "$(dirname "$GPU_LOCK")"
exec 9>"$RUNS/supervisor.lock"
flock -n 9 || exit 0
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$RUNS/supervisor.log"; }

declare -A failures
for ((chunk = 1; chunk <= MAX_CHUNKS; chunk++)); do
  line=""
  while read -r name args; do
    [[ -z $name || $name == \#* ]] && continue
    [[ -e $RUNS/$name/DONE || ${failures[$name]:-0} -ge $MAX_FAILURES ]] && continue
    line="$name $args"
    break
  done < "$RUNS/queue.txt"
  if [[ -z $line ]]; then log "queue complete"; exit 0; fi
  read -r name args <<< "$line"
  log "chunk $chunk: $name"
  # shellcheck disable=SC2086
  flock -w 7200 "$GPU_LOCK" timeout 5400 \
    uv run --extra cuda python train.py --name "$name" --data "$DATA" --runs "$RUNS" $args \
    >> "$RUNS/$name.out" 2>&1
  status=$?
  if [[ $status -eq 0 ]]; then
    log "$name complete"
  elif [[ $status -eq 75 ]]; then
    if tail -n 1 "$RUNS/$name/log.jsonl" | grep -q '"yield"'; then
      log "$name yielded the GPU; waiting"
      sleep 300
    fi
  else
    failures[$name]=$(( ${failures[$name]:-0} + 1 ))
    log "$name failed with status $status (${failures[$name]} of $MAX_FAILURES)"
  fi
  sleep 60  # the gap in which other jobs waiting on the GPU lock take it
done
log "chunk limit reached"
