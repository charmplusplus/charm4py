#!/bin/bash
# vLLM's OpenAI server with data parallelism on the Ray backend: each engine
# replica is a Ray actor (EngineCoreActor) that creates its own RayExecutorV2
# and worker actors inside a placement group. Sends two completions, then stops
# the server.
#
#   tests/ray_compat/vllm_serve_dp.sh charm4py   # under charm4py.ray.launch
#   tests/ray_compat/vllm_serve_dp.sh ray        # real Ray (starts a head)
#
# Environment: PYTHON (the venv's python), LCRUN; on a machine without GPUs set
# VLLM_CPU_RAY=1 (tests/ray_compat/vllm_plugin) and VLLM_CPU_KVCACHE_SPACE=1.
set -u
# On one machine both backends register the node as 127.0.0.1, but vLLM takes
# the default-route address as the DP master and then cannot find that node.
export VLLM_HOST_IP=${VLLM_HOST_IP:-127.0.0.1}
BACKEND=${1:-charm4py}
PY=${PYTHON:-python}
PORT=${PORT:-8765}
DP=${DP:-2}
LOG=${LOG:-/tmp/vllm_serve_dp_$BACKEND.log}
ARGS=(--model facebook/opt-125m --port $PORT --data-parallel-size $DP
      --data-parallel-backend ray --distributed-executor-backend ray
      --enforce-eager --max-model-len 256)

if [ "$BACKEND" = charm4py ]; then
  $PY -m charmrun.start +p4 -m charm4py.ray.launch --num-gpus $DP \
      -m vllm.entrypoints.openai.api_server "${ARGS[@]}" > $LOG 2>&1 &
else
  RAY=$(dirname $PY)/ray
  $RAY start --head --num-cpus 4 --num-gpus $DP --include-dashboard=false > /dev/null
  RAY_ADDRESS=auto $PY -m vllm.entrypoints.openai.api_server "${ARGS[@]}" > $LOG 2>&1 &
fi
SERVER=$!

for i in $(seq 1 240); do
  curl -sf localhost:$PORT/health > /dev/null && break
  kill -0 $SERVER 2> /dev/null || break
  sleep 1
done
STATUS=FAIL
OUT=""
if curl -sf localhost:$PORT/health > /dev/null; then
  OUT=$(for p in "Hello, my name is" "The capital of France is"; do
    curl -s localhost:$PORT/v1/completions -H 'Content-Type: application/json' \
      -d "{\"model\": \"facebook/opt-125m\", \"prompt\": \"$p\", \"max_tokens\": 16, \"temperature\": 0}" \
      | $PY -c 'import json,sys; print(repr(json.load(sys.stdin)["choices"][0]["text"]))'
  done)
  [ $(echo "$OUT" | grep -c "'") = 2 ] && STATUS=PASS
fi
kill -INT $SERVER 2> /dev/null
for i in $(seq 1 60); do kill -0 $SERVER 2> /dev/null || break; sleep 1; done
kill -9 $SERVER 2> /dev/null
[ "$BACKEND" = ray ] && $(dirname $PY)/ray stop > /dev/null 2>&1
echo "$OUT"
echo "$STATUS vllm serve, DP=$DP, $BACKEND (log: $LOG)"
