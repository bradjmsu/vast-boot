#!/usr/bin/env bash
# Registry-free boot for a vast.ai vLLM rental.
#
# This script is fetched and sha256-verified by the provision onstart bootstrap
# (see routes/llm_backends.py) before it runs. It replaces the retired baked
# Docker image: instead of pulling a 50GB image with weights inside, we boot the
# official public vLLM image and download the FP8 weights here, at boot.
#
# The dead-man watchdog (deadman.py, fetched alongside this file) is the
# supervisor and PID of the served process tree: it spawns vLLM as a child,
# monitors it, and destroys THIS vast instance the moment the server dies,
# never becomes healthy, goes idle past IDLE_MINUTES, or lives past TTL_HOURS.
# Running vLLM directly with the watchdog backgrounded would be unsafe: a vLLM
# crash would leave a stopped-but-still-billing rental behind.
set -euo pipefail

if [[ -z "${VLLM_API_KEY:-}" ]]; then
    echo "onstart: VLLM_API_KEY is not set. Refusing to start an open inference port with no key." >&2
    exit 1
fi

# hf_transfer is the accelerated HuggingFace downloader; the boot weight pull is
# the long pole, so turn it on. The official vLLM image bundles huggingface_hub
# but not always hf_transfer, so install it into the bundled hub if absent.
export HF_HUB_ENABLE_HF_TRANSFER=1
python3 -c 'import hf_transfer' 2>/dev/null || \
    pip install --no-cache-dir --break-system-packages hf_transfer

MODEL_REPO="${VLLM_HF_MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
MODEL_PATH="${VLLM_MODEL_PATH:-/models/qwen}"

# Use the image's bundled huggingface_hub CLI. Newer hub ships `hf`; older ships
# `huggingface-cli`. Both take the same `download <repo> --local-dir` form.
HF_CLI="$(command -v hf || command -v huggingface-cli)"
if [[ -z "${HF_CLI}" ]]; then
    echo "onstart: no huggingface_hub CLI (hf / huggingface-cli) found in the image." >&2
    exit 1
fi

mkdir -p "${MODEL_PATH}"
"${HF_CLI}" download "${MODEL_REPO}" --local-dir "${MODEL_PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Hand the exact retired-baked-entrypoint vLLM command to the watchdog, which
# spawns and supervises it. CUDA graphs stay on (vLLM default); --enforce-eager
# is intentionally NOT passed.
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
