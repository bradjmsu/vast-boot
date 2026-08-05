# Vast inference boot

This directory is the canonical source for the scripts that boot a Vast burst
GPU serving `Qwen/Qwen3.6-35B-A3B-FP8` through vLLM.

- `onstart.sh` downloads the model and starts vLLM under the watchdog.
- `deadman.py` supervises vLLM and destroys the rental after boot failure,
  sustained idle time, a terminal vLLM failure, or the hard TTL.
- `known-good.json` publishes the exact image digest and model revisions.

The rented box is an inference endpoint only. It does not run Prefect workers,
mount production storage, join the Lazio tailnet, or copy code from another
host. Production CPU workers consume the governed backend route and call vLLM
inline. The gateway owns rent, launch-contract persistence, readiness, model
identity, route membership, concurrency, drain, and destroy.

## Public boot artifacts

The artifacts are mirrored to the public `bradjmsu/vast-boot` repository. Vast
profiles fetch an immutable commit SHA and verify every file against the
checksums in `routes/llm_backends.py` before execution. Provisioning also
fetches and verifies those artifacts before it performs the paid create call.
The public repository is the repeatable launch artifact. It references an
immutable public vLLM image and immutable Hugging Face model revisions, so a
fresh rental does not depend on a private registry or a mutable model branch.

## Updating the scripts

1. Edit the canonical files in this directory.
2. Run `scripts/push-vast-boot.sh` from the repository root.
3. Copy the printed commit SHA and checksums into
   `services/agent-gateway/src/agent_gateway/routes/llm_backends.py`.
4. Run `scripts/pytest-file.sh tests/test_vast_burst_profiles.py`.

The checksum test fails if canonical source and the pinned public artifacts
drift.
