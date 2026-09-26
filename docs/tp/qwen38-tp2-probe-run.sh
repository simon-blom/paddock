#!/usr/bin/env bash
set -euo pipefail

# Two-node TP=2 probe launcher (rank 0 / head side). Runs ONE selected probe
# across both Sparks and collects logs plus a summary; no serving path is
# touched. Worker first (rank 1 over SSH), then rank 0 - matching the
# probes' own bootstrap order.
#
#   PROBE=abc|span   which probe to run (default abc, the current B-vs-C
#                    divergence trace; span is the A-vs-B two-rank probe)
#   BUILD=1          build the selected probe on rank 0 before launching
#   TEST_MODE        span only: "one-row" (1/16-row cases, PADDOCK_PROBE_ONE_ROW)
#                    or "normal" (full case list; default one-row)
#   DRY_RUN=1        print the exact plan (build, copy, hashes, rank
#                    commands) and exit without touching the GPU or SSH
#   RUN_DIR          override the timestamped run directory root
#                    (default /tmp/paddock-tp-span-probe/<UTC timestamp>)
#
# Loads docs/tp/qwen38-tp2-two-node.env (RUNNER/MODEL/PACK/REMOTE_*/SSH/
# NCCL/RoCE knobs). The selected release probe binary is copied to rank 1
# and its SHA256 verified on both sides before anything launches; ranks
# share one build, so a hash mismatch aborts instead of risking a
# protocol/numerics mismatch. On exit (any path) only the exact rank
# processes started by this script are cleaned up; logs and summary are
# always written.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ENV_FILE="$SCRIPT_DIR/qwen38-tp2-two-node.env"
ENV_FILE="${TP_ENV_FILE:-$DEFAULT_ENV_FILE}"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

PROBE="${PROBE:-abc}"
BUILD="${BUILD:-0}"
TEST_MODE="${TEST_MODE:-one-row}"
TP_DRY_RUN="${DRY_RUN:-${TP_DRY_RUN:-0}}"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
RUN_ROOT="${RUN_DIR:-/tmp/paddock-tp-span-probe}"
RUN_DIR="$RUN_ROOT/$STAMP"

case "$PROBE" in
  span) PROBE_BIN="qwen35_tp_span_probe" ;;
  abc) PROBE_BIN="qwen35_tp_abc_probe" ;;
  *)
    printf 'PROBE must be span or abc (got %s)\n' "$PROBE" >&2
    exit 2
    ;;
esac
case "$TEST_MODE" in
  one-row|normal) ;;
  *)
    printf 'TEST_MODE must be one-row or normal (got %s)\n' "$TEST_MODE" >&2
    exit 2
    ;;
esac

# --- config (same dotenv discipline as qwen38-tp2-two-node.sh) -------------
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
      exit 2
    fi
  done < "$file"
}

if [[ -n "${TP_ENV_FILE:-}" && ! -f "$ENV_FILE" ]]; then
  printf 'TP_ENV_FILE does not exist: %s\n' "$ENV_FILE" >&2
  exit 2
fi
load_env_file "$ENV_FILE"

: "${MODEL:?MODEL must be set (env file or environment)}"
: "${PACK:?PACK must be set}"
: "${REMOTE_MODEL:=$MODEL}"
: "${REMOTE_PACK:=$PACK}"
: "${WORKER_HOST:=192.168.100.11}"
: "${MASTER_ADDR:=192.168.100.10}"
: "${MASTER_PORT:=11566}"
: "${SSH_OPTS:=-o BatchMode=yes}"
: "${NCCL_SOCKET_IFNAME:=enp1s0f0np0}"
: "${NCCL_IB_HCA:=rocep1s0f0}"
: "${NCCL_IB_DISABLE:=0}"
: "${NCCL_NET:=IB}"
: "${REMOTE_PROBE_DIR:=/tmp/paddock-tp-probe-bin}"
: "${REMOTE_TIMEOUT:=900}"
: "${LD_LIBRARY_PATH:=}"

mkdir -p "$RUN_DIR"

# --- summary scaffolding: always written, failure paths included -----------
SUMMARY="$RUN_DIR/summary.txt"
R0_LOG="$RUN_DIR/rank0.log"
R1_LOG="$RUN_DIR/rank1.log"
: > "$SUMMARY"
: > "$R0_LOG"
: > "$R1_LOG"

note() { printf '%s\n' "$*" | tee -a "$SUMMARY" >&2; }
emit() { printf '%s\n' "$*" >> "$SUMMARY"; }

fail() {
  emit "FAIL: $*"
  finish_summary
  exit 1
}

sha256_of() { sha256sum "$1" | awk '{print $1}'; }

finish_summary() {
  emit ""
  emit "logs: $R0_LOG $R1_LOG"
}

HEAD_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"

# --- build (rank 0 only; rank 1 receives the exact binary) -----------------
LOCAL_BIN="$REPO_ROOT/target/release/examples/$PROBE_BIN"
if [[ "$BUILD" == 1 && "$TP_DRY_RUN" != 1 ]]; then
  note "building release probe: cargo build -p paddock-engine --release --example $PROBE_BIN"
  if ! cargo build -p paddock-engine --release --example "$PROBE_BIN" >> "$R0_LOG" 2>&1; then
    fail "rank0 probe build failed (see rank0.log)"
  fi
fi

# --- plan values ------------------------------------------------------------
RANK0_ENV_PREFIX="PADDOCK_TP_ABC_TRACE=1 NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME NCCL_IB_HCA=$NCCL_IB_HCA NCCL_IB_DISABLE=$NCCL_IB_DISABLE NCCL_NET=$NCCL_NET"
RANK1_ENV_PREFIX="PADDOCK_TP_ABC_TRACE=1 NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME NCCL_IB_HCA=$NCCL_IB_HCA NCCL_IB_DISABLE=$NCCL_IB_DISABLE NCCL_NET=$NCCL_NET"
if [[ "$PROBE" == "span" ]]; then
  RANK0_ENV_PREFIX="NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME NCCL_IB_HCA=$NCCL_IB_HCA NCCL_IB_DISABLE=$NCCL_IB_DISABLE NCCL_NET=$NCCL_NET"
  RANK1_ENV_PREFIX="$RANK0_ENV_PREFIX"
  if [[ "$TEST_MODE" == "one-row" ]]; then
    RANK0_ENV_PREFIX="PADDOCK_PROBE_ONE_ROW=1 $RANK0_ENV_PREFIX"
    RANK1_ENV_PREFIX="PADDOCK_PROBE_ONE_ROW=1 $RANK1_ENV_PREFIX"
  fi
fi
[[ -n "$LD_LIBRARY_PATH" ]] && {
  RANK0_ENV_PREFIX="LD_LIBRARY_PATH=$LD_LIBRARY_PATH $RANK0_ENV_PREFIX"
  RANK1_ENV_PREFIX="LD_LIBRARY_PATH=$LD_LIBRARY_PATH $RANK1_ENV_PREFIX"
}

REMOTE_BIN="$REMOTE_PROBE_DIR/$PROBE_BIN"
read -r -a SSH_ARGS <<< "$SSH_OPTS"

emit "git HEAD: $HEAD_SHA"
emit "probe: $PROBE ($PROBE_BIN)"
emit "test mode: $TEST_MODE"
emit "env file: $ENV_FILE"
emit "model: $MODEL"
emit "pack: $PACK"
emit "remote model: $REMOTE_MODEL"
emit "remote pack: $REMOTE_PACK"
emit "worker host: $WORKER_HOST master: $MASTER_ADDR:$MASTER_PORT"
emit "run dir: $RUN_DIR"

if [[ "$TP_DRY_RUN" == 1 ]]; then
  emit ""
  emit "DRY RUN - nothing executed"
  emit "build: cargo build -p paddock-engine --release --example $PROBE_BIN (BUILD=$BUILD)"
  emit "copy:  scp $LOCAL_BIN $WORKER_HOST:$REMOTE_BIN"
  emit "verify: sha256sum on both sides must match"
  emit "rank1: ssh ${SSH_ARGS[*]} $WORKER_HOST -- \"env $RANK1_ENV_PREFIX $REMOTE_BIN 1 $MASTER_ADDR $REMOTE_MODEL $REMOTE_PACK $MASTER_PORT\" > $R1_LOG"
  emit "rank0: env $RANK0_ENV_PREFIX $LOCAL_BIN 0 $MASTER_ADDR $MODEL $PACK $MASTER_PORT > $R0_LOG"
  finish_summary
  printf 'summary: %s\n' "$SUMMARY"
  exit 0
fi

[[ -x "$LOCAL_BIN" ]] || fail "probe binary missing; run with BUILD=1 ($LOCAL_BIN)"
[[ -f "$MODEL" ]] || fail "MODEL does not exist: $MODEL"
[[ -f "$PACK" ]] || fail "PACK does not exist: $PACK"

# --- stage + verify the binary on rank 1 ------------------------------------
note "staging probe binary on $WORKER_HOST:$REMOTE_BIN"
ssh "${SSH_ARGS[@]}" "$WORKER_HOST" "mkdir -p $(printf '%q' "$REMOTE_PROBE_DIR")" \
  || fail "cannot create $REMOTE_PROBE_DIR on $WORKER_HOST"
scp -q "$LOCAL_BIN" "$WORKER_HOST:$REMOTE_BIN" || fail "scp to $WORKER_HOST failed"
LOCAL_SHA="$(sha256_of "$LOCAL_BIN")"
REMOTE_SHA="$(ssh "${SSH_ARGS[@]}" "$WORKER_HOST" "sha256sum $(printf '%q' "$REMOTE_BIN")" | awk '{print $1}')"
emit "binary sha256: $LOCAL_SHA"
[[ "$LOCAL_SHA" == "$REMOTE_SHA" ]] || fail "binary hash mismatch rank0=$LOCAL_SHA rank1=$REMOTE_SHA"
[[ -f "$REMOTE_MODEL" ]] || ssh "${SSH_ARGS[@]}" "$WORKER_HOST" "test -f $(printf '%q' "$REMOTE_MODEL")" \
  || fail "REMOTE_MODEL not reachable on $WORKER_HOST: $REMOTE_MODEL"
ssh "${SSH_ARGS[@]}" "$WORKER_HOST" "test -f $(printf '%q' "$REMOTE_PACK")" \
  || fail "REMOTE_PACK not reachable on $WORKER_HOST: $REMOTE_PACK"

# --- run: worker first, then rank 0 -----------------------------------------
# Both probes are one-shot: rank 1 exits when rank 0's Shutdown (or NCCL
# abort) lands, and the remote command is wrapped in `timeout` so a hung
# rank can never outlive REMOTE_TIMEOUT seconds. Cleanup kills only the
# exact processes this script started: the local rank-1 ssh PID and, on the
# worker, the timeout-wrapped probe it launched - never a pkill sweep.
RANK1_PID=""
RANK1_RC=0
remote_cleanup() {
  set +e
  if [[ -n "$RANK1_PID" ]] && kill -0 "$RANK1_PID" 2>/dev/null; then
    kill -TERM "$RANK1_PID" 2>/dev/null
    local waited=0
    while kill -0 "$RANK1_PID" 2>/dev/null && (( waited < 15 )); do
      sleep 1
      waited=$((waited + 1))
    done
    kill -KILL "$RANK1_PID" 2>/dev/null
  fi
}
trap remote_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

note "launching rank 1 on $WORKER_HOST"
ssh "${SSH_ARGS[@]}" "$WORKER_HOST" \
  "env $RANK1_ENV_PREFIX timeout ${REMOTE_TIMEOUT} $(printf '%q' "$REMOTE_BIN") 1 $(printf '%q' "$MASTER_ADDR") $(printf '%q' "$REMOTE_MODEL") $(printf '%q' "$REMOTE_PACK") $(printf '%q' "$MASTER_PORT")" \
  > "$R1_LOG" 2>&1 &
RANK1_PID=$!

note "launching rank 0 locally"
set +e
env $RANK0_ENV_PREFIX "$LOCAL_BIN" 0 "$MASTER_ADDR" "$MODEL" "$PACK" "$MASTER_PORT" \
  > "$R0_LOG" 2>&1
RANK0_RC=$?
set -e

# Rank 1 exits when rank 0's shutdown lands; give it a bounded grace window
# before escalating (TERM, then KILL via the EXIT trap's remote_cleanup).
waited=0
while kill -0 "$RANK1_PID" 2>/dev/null; do
  (( waited >= 30 )) && break
  sleep 1
  waited=$((waited + 1))
done
remote_cleanup
# reap_rank1: wait must run while RANK1_PID is still set, or `wait ""`
# errors with status 1 and the summary would report a bogus rank1 exit.
RANK1_RC=0
wait "$RANK1_PID" 2>/dev/null || RANK1_RC=$?
RANK1_PID=""

# --- summarize ---------------------------------------------------------------
emit "rank0 exit: $RANK0_RC"
emit "rank1 exit: $RANK1_RC"
INFRA="PASS"
[[ "$RANK0_RC" != 0 || "$RANK1_RC" != 0 ]] && INFRA="FAIL"
emit "infrastructure: $INFRA (rank exit codes only - does not imply the comparison ran)"

# Numerical-comparison verdict, parsed from the probe's own stdout rather
# than inferred from process exits: a zero exit with an empty arm trace is
# an infrastructure pass AND a comparison failure at once (the c_stages=0
# run summarized as PASS on exits alone - the failure mode this fixes).
COMPARISON="INVALID: no probe verdict found in rank0 log"
if [[ "$PROBE" == "abc" ]]; then
  if grep -qE '^abc_probe .* b_stages=0( |$)' "$R0_LOG"; then
    COMPARISON="INVALID: B arm traced zero stages"
  elif grep -qE '^abc_probe .* c_stages=0( |$)' "$R0_LOG"; then
    COMPARISON="INVALID: C arm traced zero stages"
  elif grep -q "first material divergence" "$R0_LOG"; then
    COMPARISON="DIVERGENCE ($(grep -m1 'first material divergence' "$R0_LOG" | sed 's/^first material divergence: //'))"
  elif grep -qE "compared_stage_pairs=0" "$R0_LOG"; then
    COMPARISON="INVALID: zero comparable stage pairs"
  elif grep -q "no stage exceeded" "$R0_LOG"; then
    COMPARISON="MATCH (all compared stages within threshold)"
  fi
else
  if grep -q "VIOLATION" "$R0_LOG"; then
    COMPARISON="DIVERGENCE ($(grep -m1 'VIOLATION' "$R0_LOG" | sed 's/.*span_probe //'))"
  elif grep -q "span_probe OK" "$R0_LOG"; then
    COMPARISON="MATCH (all cases within tolerance)"
  fi
fi
emit "comparison: $COMPARISON"
if [[ "$INFRA" == "PASS" && "$COMPARISON" != INVALID* ]]; then
  # A valid run: MATCH or a located DIVERGENCE both mean the diagnostic
  # itself succeeded; the comparison line above carries the verdict.
  emit "result: PASS"
else
  emit "result: FAIL"
fi
emit ""
emit "--- probe result lines (rank0) ---"
grep -E "span_probe|abc_probe|compared_stage_pairs|unmatched_[bc]_stages|first material divergence|no stage exceeded|VIOLATION|max_abs" "$R0_LOG" >> "$SUMMARY" 2>/dev/null || true
if [[ "$PROBE" == "abc" ]]; then
  emit ""
  emit "--- B-C per-layer trace (rank0) ---"
  grep -E "^layer +[0-9]+ |final-norm|first material divergence|no stage exceeded|compared_stage_pairs|unmatched_[bc]_stages" "$R0_LOG" >> "$SUMMARY" 2>/dev/null || true
fi
emit ""
emit "--- WARN/ERROR/VIOLATION lines ---"
grep -hE "WARN|ERROR|VIOLATION|panic|Error" "$R0_LOG" "$R1_LOG" >> "$SUMMARY" 2>/dev/null || true
finish_summary

printf 'summary: %s\n' "$SUMMARY"
printf 'rank0 log: %s\n' "$R0_LOG"
printf 'rank1 log: %s\n' "$R1_LOG"
