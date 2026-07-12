# vast-boot

Registry-free boot scripts for renting burst GPUs on vast.ai to serve the FP8
`Qwen/Qwen3.6-35B-A3B-FP8` model with vLLM. This repo holds exactly two runtime
files plus this README:

- `onstart.sh` : the boot script. Downloads the model weights at boot (via
  HuggingFace with `hf_transfer` acceleration), then hands vLLM to the dead-man
  watchdog which supervises it.
- `deadman.py` : the dead-man watchdog. Standard library only. It is the
  supervisor: it spawns vLLM as a child, monitors it, and destroys its own
  vast.ai instance the moment vLLM dies, never becomes healthy, sits idle past
  `IDLE_MINUTES`, or lives past `TTL_HOURS`. A forgotten rental can never keep
  billing forever.

## Why this repo exists (the pivot)

The earlier approach baked the weights and vLLM into a private Docker image and
pushed it to a registry. That was retired on 2026-07-12: registry auth was a
recurring pain, and the storage plus egress economics of shipping a ~50GB image
were not worth it. Downloading the weights at boot on a 1Gbps host is
comparable-or-faster than pulling a 50GB image, and it needs no registry.

Now a rented box boots the official public vLLM image and runs these two scripts.

## This repo holds NO secrets

`onstart.sh` and `deadman.py` are safe to be public. Every secret (the vLLM
serve key `VLLM_API_KEY`, the vast destroy key `VAST_DESTROY_KEY`, the instance
id) is injected into the rented container as an environment variable by the
backend at provision time. Nothing sensitive lives here.

## How the backend uses these files (SHA-pinned)

The Lazio backend does NOT fetch these scripts from a branch. Its vast deploy
profiles pin the fetch to a specific **commit SHA** of this repo
(`raw.githubusercontent.com/Lazio-Partners/vast-boot/<COMMIT_SHA>/...`) and
verify each downloaded file against a baked-in `sha256` checksum before running
it. Pinning by SHA (never a moving branch ref) plus the checksum check means a
compromise of this repo cannot change what already-rented boxes execute, and a
change to the boot scripts is only picked up by a deliberate, reviewed backend
bump.

## Canonical source lives in the backend

These files are mirrored here from the `Lazio-Partners/backend` repo
(`services/vast-image/deadman.py` and `services/vast-image/onstart.sh`). Do not
hand-edit them here. Change them in the backend, then run
`scripts/push-vast-boot.sh` there, which syncs both files to this repo and
prints the new commit SHA and checksums to paste into
`routes/llm_backends.py`.
