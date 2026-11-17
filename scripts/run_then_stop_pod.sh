#!/usr/bin/env bash
# Run a long job on a RunPod pod, then stop the pod however the job ends.
#
#   scripts/run_then_stop_pod.sh [--max-hours H] [--log FILE] -- COMMAND [ARG...]
#   scripts/run_then_stop_pod.sh --check
#
# A pod bills for its GPU until someone stops it, so a training run that outlasts the
# session that launched it keeps billing after it finishes or crashes. This wrapper:
#
#   - refuses to start unless it can find runpodctl and this pod's id, so it never starts
#     a job it could not stop afterwards (--check does only this test);
#   - runs COMMAND with its output appended to FILE as it is produced: Python is set
#     unbuffered and nothing sits between the job and the file;
#   - kills COMMAND after H hours, if --max-hours is given;
#   - then stops the pod: after success, after failure, at the deadline, or when the
#     wrapper itself is interrupted. A failed stop call is retried for half an hour.
#
# It exits with COMMAND's exit status.
#
# Stopping releases the GPU and CLEARS THE CONTAINER DISK. Only a volume disk survives, so
# create the pod with one mounted at /workspace and keep the repository, the model cache
# and data/adapters under it. A stopped pod still bills a little for that disk, and can
# only be started again when its host has a free GPU, so copy results off while it runs
# when you can, and terminate the pod once you have them.
#
# The pod id and API key belong to the container's main process and are missing from SSH
# sessions, so they are read from /proc/1/environ when the environment lacks them. Neither
# is ever printed.

set -u

RUNPODCTL=${RUNPODCTL:-runpodctl}
PID1_ENVIRON=${PID1_ENVIRON:-/proc/1/environ}
STOP_ATTEMPTS=${STOP_ATTEMPTS:-30}
STOP_RETRY_SECONDS=${STOP_RETRY_SECONDS:-60}

usage() {
  sed -n '2,5p' "$0" | sed 's/^# \{0,1\}//' >&2
  exit 2
}

max_hours=""
log="job.log"
check=0
while [ $# -gt 0 ]; do
  case "$1" in
    --max-hours) [ $# -ge 2 ] || usage; max_hours=$2; shift 2 ;;
    --log) [ $# -ge 2 ] || usage; log=$2; shift 2 ;;
    --check) check=1; shift ;;
    --) shift; break ;;
    *) usage ;;
  esac
done

# The value of one environment variable of the container's main process, or nothing.
from_main_process() {
  [ -r "$PID1_ENVIRON" ] || return 0
  tr '\0' '\n' < "$PID1_ENVIRON" | sed -n "s/^$1=//p" | head -n 1
}

if [ -z "${RUNPOD_POD_ID:-}" ]; then
  RUNPOD_POD_ID=$(from_main_process RUNPOD_POD_ID)
fi
if [ -z "${RUNPOD_API_KEY:-}" ]; then
  RUNPOD_API_KEY=$(from_main_process RUNPOD_API_KEY)
fi
export RUNPOD_POD_ID
[ -n "$RUNPOD_API_KEY" ] && export RUNPOD_API_KEY

problems=0
if ! command -v "$RUNPODCTL" > /dev/null 2>&1; then
  echo "run_then_stop_pod: $RUNPODCTL not found" >&2
  problems=1
fi
if [ -z "$RUNPOD_POD_ID" ]; then
  echo "run_then_stop_pod: no pod id in RUNPOD_POD_ID or $PID1_ENVIRON" >&2
  problems=1
fi
if [ "$check" = 1 ]; then
  if [ "$problems" = 0 ]; then
    key="no API key in the environment"
    [ -n "$RUNPOD_API_KEY" ] && key="API key present"
    echo "ok: $RUNPODCTL found, pod id known, $key"
  fi
  exit "$problems"
fi
if [ "$problems" != 0 ]; then
  echo "run_then_stop_pod: not starting a job that could not be stopped afterwards" >&2
  exit 2
fi
[ $# -gt 0 ] || usage

say() {
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$log"
}

# Ask RunPod to stop this pod. Tries the current CLI syntax, then the older one.
stop_pod() {
  trap - EXIT INT TERM
  attempt=1
  while [ "$attempt" -le "$STOP_ATTEMPTS" ]; do
    say "stopping this pod (attempt $attempt of $STOP_ATTEMPTS)"
    if "$RUNPODCTL" pod stop "$RUNPOD_POD_ID" >> "$log" 2>&1 \
      || "$RUNPODCTL" stop pod "$RUNPOD_POD_ID" >> "$log" 2>&1; then
      say "stop requested"
      return 0
    fi
    attempt=$((attempt + 1))
    sleep "$STOP_RETRY_SECONDS"
  done
  say "COULD NOT STOP THE POD: it is still running and billing"
  return 1
}

export PYTHONUNBUFFERED=1
# Job control puts the job in its own process group, so the whole tree can be killed.
set -m
say "start: $*"
"$@" >> "$log" 2>&1 < /dev/null &
job=$!

watchdog=""
if [ -n "$max_hours" ]; then
  seconds=$(awk -v hours="$max_hours" 'BEGIN { s = int(hours * 3600); print (s < 1 ? 1 : s) }')
  (
    sleep "$seconds"
    echo "=== deadline of $max_hours hours reached: killing the job" >> "$log"
    kill -TERM -- "-$job" 2> /dev/null
    sleep 20
    kill -KILL -- "-$job" 2> /dev/null
  ) > /dev/null 2>&1 &
  watchdog=$!
fi

trap 'say "interrupted"; kill -TERM -- "-$job" 2> /dev/null; exit 130' INT TERM
trap stop_pod EXIT

wait "$job"
status=$?
[ -n "$watchdog" ] && kill -- "-$watchdog" 2> /dev/null
say "job ended with exit status $status"
exit "$status"
