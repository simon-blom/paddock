#!/usr/bin/env bash
set -euo pipefail

# Run this on rank 0/head. The worker is started over SSH with --tp-worker.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ENV_FILE="$SCRIPT_DIR/qwen38-tp2-two-node.env"
ENV_FILE="${TP_ENV_FILE:-$DEFAULT_ENV_FILE}"

CONFIG_KEYS=(
  RUNNER REMOTE_RUNNER MODEL REMOTE_MODEL PACK REMOTE_PACK
  WORKER_HOST MASTER_ADDR MASTER_PORT HTTP_HOST HTTP_PORT MAX_CTX MAX_BATCH
  KV_DTYPE SPEC TP_GRAPH SSH_OPTS NCCL_SOCKET_IFNAME NCCL_IB_HCA
  NCCL_IB_DISABLE NCCL_NET REMOTE_PIDFILE REMOTE_LOG LD_LIBRARY_PATH
)

# Read simple KEY=VALUE dotenv files without executing them. Caller-provided
# environment values win: only variables that are currently unset are loaded.
load_env_file() {
  local file="$1" line key value
  [[ -f "$file" ]] || return 0
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    [[ -z "$line" || "${line:0:1}" == "#" ]] && continue
    if [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
      key="${BASH_REMATCH[1]}"
      value="${BASH_REMATCH[2]}"
      value="${value%"${value##*[![:space:]]}"}"
      if [[ ${#value} -ge 2 && ( ( "${value:0:1}" == '"' && "${value: -1}" == '"' ) || ( "${value:0:1}" == "'" && "${value: -1}" == "'" ) ) ]]; then
        value="${value:1:${#value}-2}"
      fi
      if [[ ! -v "$key" ]]; then
        printf -v "$key" '%s' "$value"
        export "$key"
      fi
    else
      printf 'invalid dotenv line in %s: %s\n' "$file" "$line" >&2
      return 2
    fi
  done < "$file"
}

if [[ -n "${TP_ENV_FILE:-}" && ! -f "$ENV_FILE" ]]; then
  printf 'TP_ENV_FILE does not exist: %s\n' "$ENV_FILE" >&2
  exit 2
fi
load_env_file "$ENV_FILE"

: "${RUNNER:=./target/release/paddock-runner}"
: "${MODEL:=/path/to/Qwen3.8-27B-UD-Q4_K_M.gguf}"
: "${PACK:=./packs/cuda/build/pd-cuda-sm120.so}"
: "${REMOTE_RUNNER:=$RUNNER}"
: "${REMOTE_MODEL:=$MODEL}"
: "${REMOTE_PACK:=$PACK}"
: "${WORKER_HOST:=192.168.100.11}"
: "${MASTER_ADDR:=192.168.100.10}"
: "${MASTER_PORT:=11560}"
: "${HTTP_HOST:=0.0.0.0}"
: "${HTTP_PORT:=11540}"
: "${MAX_CTX:=65536}"
: "${MAX_BATCH:=2}"
: "${KV_DTYPE:=f16}"
: "${SPEC:=on}"
: "${TP_GRAPH:=1}"
: "${SSH_OPTS:=}"
: "${NCCL_SOCKET_IFNAME:=enp1s0f0np0}"
: "${NCCL_IB_HCA:=rocep1s0f0}"
: "${NCCL_IB_DISABLE:=0}"
: "${NCCL_NET:=IB}"
: "${LD_LIBRARY_PATH:=}"
: "${REMOTE_PIDFILE:=/tmp/paddock-qwen38-tp2-worker-${MASTER_PORT}.pid}"
: "${REMOTE_LOG:=/tmp/paddock-qwen38-tp2-worker-${MASTER_PORT}.log}"
: "${TP_DRY_RUN:=0}"

require_nonempty() {
  local name="$1" value="${!1:-}"
  [[ -n "$value" ]] || { printf '%s must be set\n' "$name" >&2; exit 2; }
}

for name in RUNNER MODEL PACK WORKER_HOST MASTER_ADDR MASTER_PORT HTTP_PORT MAX_CTX MAX_BATCH KV_DTYPE SPEC TP_GRAPH; do
  require_nonempty "$name"
done

if [[ "$TP_DRY_RUN" != 1 ]]; then
  [[ -x "$RUNNER" ]] || { printf 'RUNNER is not executable: %s\n' "$RUNNER" >&2; exit 2; }
  [[ -f "$MODEL" ]] || { printf 'MODEL does not exist: %s\n' "$MODEL" >&2; exit 2; }
  [[ -f "$PACK" ]] || { printf 'PACK does not exist: %s\n' "$PACK" >&2; exit 2; }
fi

read -r -a SSH_ARGS <<< "$SSH_OPTS"
remote_pid=""

quote_remote() {
  printf '%q' "$1"
}

remote_cleanup() {
  local pidfile="$REMOTE_PIDFILE" pid cmdline
  set +e
  [[ -n "$remote_pid" ]] || return 0
  ssh "${SSH_ARGS[@]}" "$WORKER_HOST" bash -s -- "$pidfile" <<'REMOTE_CLEANUP' >/dev/null 2>&1
set -e
pidfile="$1"
if [[ -r "$pidfile" ]]; then
  pid="$(<"$pidfile")"
  if [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/cmdline" ]]; then
    cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    if [[ "$cmdline" == *"--tp-worker"* && "$cmdline" == *"paddock-runner"* ]]; then
      kill -TERM "$pid" 2>/dev/null || true
      for _ in {1..20}; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
      done
      kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
  rm -f "$pidfile"
fi
REMOTE_CLEANUP
}

trap remote_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

remote_worker_cmd=""
remote_env=(
  "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
  "NCCL_IB_HCA=$NCCL_IB_HCA"
  "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
  "NCCL_NET=$NCCL_NET"
)
[[ -n "$LD_LIBRARY_PATH" ]] && remote_env+=("LD_LIBRARY_PATH=$LD_LIBRARY_PATH")
for arg in env -u PADDOCK_TP_GRAPH -u PADDOCK_TP_NO_SPAWN "${remote_env[@]}" \
  "$REMOTE_RUNNER" --tp-worker \
  --model "$REMOTE_MODEL" --kernel-pack "$REMOTE_PACK" \
  --tp-master-addr "$MASTER_ADDR" --tp-master-port "$MASTER_PORT"; do
  remote_worker_cmd+=" $(quote_remote "$arg")"
done

if [[ "$TP_DRY_RUN" == 1 ]]; then
  printf 'env file: %s\n' "$ENV_FILE"
  printf 'remote worker: ssh %s %s\n' "$WORKER_HOST" "$remote_worker_cmd"
  printf 'coordinator: PADDOCK_TP_GRAPH=%q PADDOCK_TP_NO_SPAWN=1 %q' "$TP_GRAPH" "$RUNNER"
  printf ' --model %q --kernel-pack %q --device cuda --tp-size 2' "$MODEL" "$PACK"
  printf ' --tp-master-addr %q --tp-master-port %q --host %q --port %q' "$MASTER_ADDR" "$MASTER_PORT" "$HTTP_HOST" "$HTTP_PORT"
  printf ' --max-ctx %q --max-batch %q --kv-cache-dtype %q' "$MAX_CTX" "$MAX_BATCH" "$KV_DTYPE"
  if [[ "$SPEC" == off ]]; then printf ' --no-spec\n'; else printf ' --spec %q\n' "$SPEC"; fi
  exit 0
fi

# Refuse an existing live worker with the same exact pidfile instead of
# attaching to or killing an unrelated Paddock process.
ssh "${SSH_ARGS[@]}" "$WORKER_HOST" bash -s -- "$REMOTE_PIDFILE" "$REMOTE_LOG" "$remote_worker_cmd" <<'REMOTE_START'
set -euo pipefail
pidfile="$1"
logfile="$2"
shift 2
command="$*"
if [[ -r "$pidfile" ]]; then
  old="$(<"$pidfile")"
  if [[ "$old" =~ ^[0-9]+$ && -r "/proc/$old/cmdline" ]]; then
    cmdline="$(tr '\0' ' ' < "/proc/$old/cmdline" 2>/dev/null || true)"
    if [[ "$cmdline" == *"--tp-worker"* && "$cmdline" == *"paddock-runner"* ]]; then
      printf 'worker pidfile already refers to a live worker: %s\n' "$old" >&2
      exit 3
    fi
  fi
  rm -f "$pidfile"
fi
nohup bash -lc "$command" >"$logfile" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$pidfile"
for _ in {1..10}; do
  kill -0 "$pid" 2>/dev/null && { printf '%s\n' "$pid"; exit 0; }
  sleep 1
done
printf 'worker exited during startup; log: %s\n' "$logfile" >&2
cat "$logfile" >&2 || true
exit 4
REMOTE_START
remote_pid="started"

coordinator=("$RUNNER" --model "$MODEL" --kernel-pack "$PACK" --device cuda
  --tp-size 2 --tp-master-addr "$MASTER_ADDR" --tp-master-port "$MASTER_PORT"
  --host "$HTTP_HOST" --port "$HTTP_PORT" --max-ctx "$MAX_CTX"
  --max-batch "$MAX_BATCH" --kv-cache-dtype "$KV_DTYPE")
if [[ "$SPEC" == off ]]; then
  coordinator+=(--no-spec)
else
  coordinator+=(--spec "$SPEC")
fi

# Rank 0 is graph-authoritative. The worker deliberately receives no local
# PADDOCK_TP_GRAPH value; TpInit sends the resolved graph decision to rank 1.
coordinator_env=(
  "PADDOCK_TP_GRAPH=$TP_GRAPH"
  PADDOCK_TP_NO_SPAWN=1
  "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
  "NCCL_IB_HCA=$NCCL_IB_HCA"
  "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
  "NCCL_NET=$NCCL_NET"
)
[[ -n "$LD_LIBRARY_PATH" ]] && coordinator_env+=("LD_LIBRARY_PATH=$LD_LIBRARY_PATH")
env "${coordinator_env[@]}" "${coordinator[@]}"
