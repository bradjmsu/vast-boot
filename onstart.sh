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

# --- Optional on-box Prefect worker mode (issue #1249) --------------------------
# When LAZIO_VAST_ONBOX=1 the provision env (routes/llm_backends.py) also injects
# a tailscale key + Prefect worker config, and this box runs the Prefect workers
# itself so the per-run orchestration that starves the GPU moves off hermes's 8
# vCPUs onto this box's idle CPUs. The card serves 64-128 concurrent sequences;
# per-run flow orchestration on hermes had capped the drain at 350-1300 runs/hr.
#
# This whole block is best-effort and FAIL-SAFE: any failure logs loudly, turns
# on-box mode OFF, and the box still serves vLLM on its public URL for the
# hermes-side workers exactly as before (no regression). deadman.py then starts
# the workers only after vLLM is healthy AND the shared result mount is live, so
# a half-set-up box never silently drops results. The remote-boundary specifics
# (tailscale userspace networking, tailscale-proxied ssh to hermes, sshfs/FUSE)
# are the pieces that must be validated on a real rented box before flipping
# on-box mode on in production; see the PR body for the live-smoke checklist.
_onbox_setup() {
    echo "onstart: on-box worker mode requested (LAZIO_VAST_ONBOX=1); setting up"
    if [[ -z "${TAILSCALE_AUTHKEY:-}" || -z "${PREFECT_API_URL:-}" \
        || -z "${LAZIO_PREFECT_ONBOX_RESULTS_REMOTE:-}" \
        || -z "${PREFECT_LOCAL_STORAGE_PATH:-}" ]]; then
        echo "onstart: on-box env incomplete; disabling on-box mode" >&2
        return 1
    fi
    local cid="${CONTAINER_ID:-unknown}"
    local ts_dir=/opt/tailscale

    # 1) tailscale static binaries + userspace networking. Vast containers have
    #    no /dev/net/tun, so tailscaled runs with --tun=userspace-networking and
    #    exposes a local SOCKS5/HTTP proxy that the Prefect client + ssh use.
    mkdir -p "${ts_dir}" /var/lib/tailscale
    python3 - "${ts_dir}" <<'PY' || return 1
import io, os, sys, tarfile, urllib.request
dest = sys.argv[1]
ver = os.environ.get("TAILSCALE_VERSION", "1.80.3")
url = f"https://pkgs.tailscale.com/stable/tailscale_{ver}_amd64.tgz"
data = urllib.request.urlopen(url, timeout=120).read()
with tarfile.open(fileobj=io.BytesIO(data)) as tf:
    for m in tf.getmembers():
        base = os.path.basename(m.name)
        if base in ("tailscale", "tailscaled") and m.isfile():
            m.name = base
            tf.extract(m, dest)
            os.chmod(os.path.join(dest, base), 0o755)
PY
    export PATH="${ts_dir}:${PATH}"
    "${ts_dir}/tailscaled" --tun=userspace-networking \
        --socks5-server=localhost:1055 \
        --outbound-http-proxy-listen=localhost:1055 \
        --statedir=/var/lib/tailscale >/var/log/tailscaled.log 2>&1 &
    sleep 3
    "${ts_dir}/tailscale" up --authkey="${TAILSCALE_AUTHKEY}" \
        --hostname="vast-${cid}" --accept-routes --timeout=60s || return 1
    # Route the Prefect client (httpx honours these) through the userspace proxy;
    # keep localhost direct so deadman's vLLM health checks are never proxied.
    export ALL_PROXY="socks5://localhost:1055"
    export HTTP_PROXY="http://localhost:1055" HTTPS_PROXY="http://localhost:1055"
    export NO_PROXY="localhost,127.0.0.1,::1" no_proxy="localhost,127.0.0.1,::1"
    # ssh to hermes proxied through tailscale (no /dev/net/tun, so `tailscale nc`
    # is the connect path). Tailscale SSH on hermes authenticates us by tailnet
    # identity via an ACL grant for the vast tag, so NO ssh key ships on the box.
    # A wrapper script avoids ProxyCommand word-splitting when passed to rsync -e
    # / sshfs ssh_command.
    cat > "${ts_dir}/onbox-ssh" <<EOF
#!/usr/bin/env bash
exec ssh -o StrictHostKeyChecking=accept-new -o BatchMode=yes \\
    -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \\
    -o ProxyCommand="${ts_dir}/tailscale nc %h %p" "\$@"
EOF
    chmod +x "${ts_dir}/onbox-ssh"

    # 2) shared Prefect result mount. backend_call persists its result here and
    #    the hermes-side parent that called run_deployment reads it back via
    #    state.result(); a missing/dead mount means silent result loss, so this
    #    is required for on-box mode (deadman re-checks the mount before workers).
    mkdir -p "${PREFECT_LOCAL_STORAGE_PATH}"
    command -v sshfs >/dev/null 2>&1 || \
        (apt-get update -y && apt-get install -y sshfs) >/dev/null 2>&1 || return 1
    sshfs -o "reconnect,allow_other,ssh_command=${ts_dir}/onbox-ssh" \
        "${LAZIO_PREFECT_ONBOX_RESULTS_REMOTE}" "${PREFECT_LOCAL_STORAGE_PATH}" || return 1
    mountpoint -q "${PREFECT_LOCAL_STORAGE_PATH}" || return 1

    # 3) flow code + industry_graph. A process worker imports the flow locally
    #    (deployments are `prefect deploy <path>:<flow>`, no remote pull step), so
    #    the box needs the flows tree and the industry_graph package. Sync both
    #    from hermes over the same tailscale-proxied ssh (mirrors worker-home's
    #    sync-from-hermes.sh; no new credential).
    local host="${LAZIO_ONBOX_HERMES_HOST:-ubuntu@100.126.249.14}"
    local remote_root="${LAZIO_ONBOX_HERMES_ROOT:-/home/ubuntu/.hermes}"
    mkdir -p /opt/prefect/flows
    rsync -a --delete -e "${ts_dir}/onbox-ssh" \
        "${host}:${remote_root}/services/prefect/flows/" /opt/prefect/flows/ || return 1
    pip install --no-cache-dir --break-system-packages "prefect>=3,<4" || return 1
    rsync -a -e "${ts_dir}/onbox-ssh" \
        "${host}:${remote_root}/skills/lazio/industry-graph/" /opt/industry-graph/ || return 1
    pip install --no-cache-dir --break-system-packages /opt/industry-graph || return 1
    echo "onstart: on-box worker mode armed (workers start after vLLM health)"
    return 0
}

if [[ "${LAZIO_VAST_ONBOX:-0}" == "1" ]]; then
    if _onbox_setup; then
        export LAZIO_VAST_ONBOX=1
    else
        echo "onstart: on-box setup FAILED; falling back to public-URL vLLM only" >&2
        export LAZIO_VAST_ONBOX=0
    fi
fi

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
