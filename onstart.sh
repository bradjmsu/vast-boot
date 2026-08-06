#!/usr/bin/env bash
# Registry-free boot for a vast.ai vLLM rental.
#
# The provision bootstrap fetches and verifies this file before execution.
# This box runs inference only. Production CPU workers call it through the
# governed backend route; no Prefect worker, Hermes mount, or Tailscale setup
# belongs on rented GPU hardware.
set -euo pipefail

if [[ -z "${VLLM_API_KEY:-}" ]]; then
    echo "onstart: VLLM_API_KEY is not set. Refusing to start an open inference port with no key." >&2
    exit 1
fi

# The Xet CDN path repeatedly lost TLS connections on a live RTX PRO 6000
# rental and wrote only 65 MB in 15 minutes. Regular HTTP sustained more than
# 100 MB/s on the same host. Avoid Xet so cold start is fast and predictable.
export HF_HUB_DISABLE_XET=1

MODEL_REPO="${VLLM_HF_MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
MODEL_REVISION="${VLLM_HF_REVISION:?VLLM_HF_REVISION is required}"
MODEL_PATH="${VLLM_MODEL_PATH:-/models/qwen}"
HF_CLI="$(command -v hf || command -v huggingface-cli)"
if [[ -z "${HF_CLI}" ]]; then
    echo "onstart: no huggingface_hub CLI found in the image." >&2
    exit 1
fi

mkdir -p "${MODEL_PATH}"
"${HF_CLI}" download "${MODEL_REPO}" --revision "${MODEL_REVISION}" --local-dir "${MODEL_PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec python3 "${SCRIPT_DIR}/deadman.py" \
    vllm serve "${MODEL_PATH}" \
    --served-model-name qwen \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key "$VLLM_API_KEY" \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --kv-cache-dtype fp8_e4m3 \
    --calculate-kv-scales \
    --max-num-seqs 128
