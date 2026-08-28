# Agent Training Toolkit

This toolkit turns a SpeechT5 experiment into a configuration change instead of a
new orchestration script.

It separates five operations:

1. Create one secure RTX 4090 pod and persist its identity.
2. Start or reattach one named training profile on an explicit pod ID.
3. Run a shared training engine with an approach-specific entrypoint.
4. Poll provider, process, checkpoint, GPU, disk, and evaluation state.
5. Download verified checkpoints and final artifacts before pod termination.

## Working documents

Every run writes to:

```text
artifacts/stage11/agent-runs/<run-id>/
├── pod.json
├── run.json
├── status.json
├── events.jsonl
├── checkpoint-inventory.json
├── checkpoints/
└── final/
```

`pod.json` is the provider lifecycle record. It contains the pod ID, selected
GPU, security/interruptibility settings, hourly price, SSH endpoint, and
termination time.

`run.json` is the agent handoff document. It contains the approach, config and
model revisions, dataset location, bundle hash, remote command/PID, current
phase, artifact verification state, and paths required to resume control.

The RunPod API key remains in the repository `.env`. Its value is never copied
into a working document, command line, training bundle, or remote pod.

## Available approaches

| Profile | Module | Training behavior |
|---|---|---|
| `v1a-conservative-full` | `training.v1_approaches.conservative_full` | Full model, low learning rate |
| `v1b-lora` | `training.v1_approaches.lora` | Frozen base plus LoRA attention adapters |
| `v1c-gradual-unfreeze` | `training.v1_approaches.gradual_unfreeze` | Modal heads first, then top two decoder blocks |
| `v1d-reduction-factor-1` | `training.v1_approaches.reduction_factor_1` | Reduction factor 1 with compatible output heads |

All profiles point to the same locked V1 manifests. Model, vocoder, speaker
encoder, and ASR revisions remain pinned in YAML.

## Agent workflow

Run commands from `D:\OpenVoice-Lab`:

```powershell
$python = 'D:\OpenVoice-Lab\backend\.venv\Scripts\python.exe'
$runId = 'v1a-20260828-001'

& $python -m training.runpod_agent.create_pod `
  --repository-root 'D:\OpenVoice-Lab' `
  --run-id $runId `
  --approach v1a-conservative-full
```

The command prints and saves the pod ID. Use that exact ID for launch:

```powershell
$run = "D:\OpenVoice-Lab\artifacts\stage11\agent-runs\$runId\run.json"
$podId = (Get-Content -Raw $run | ConvertFrom-Json).pod_id

& $python -m training.runpod_agent.start_training `
  --run-document $run `
  --pod-id $podId `
  --data-root 'D:\OpenVoice-Lab'
```

The launcher requires a clean committed worktree, validates the config and
dataset lock, records the source commit, creates a SHA-256 bundle,
waits for SSH, verifies the upload, installs pinned dependencies, and records
the remote PID. Calling it again with the same pod/run reattaches or resumes
from the latest valid checkpoint.

Start the durable watcher:

```powershell
& $python -m training.runpod_agent.watch `
  --run-document $run `
  --poll-seconds 20 `
  --terminate-on-complete
```

The watcher repeatedly:

- refreshes `status.json`;
- captures optimizer step, evaluation progress, GPU telemetry, and disk state;
- detects checkpoint completion markers;
- downloads each checkpoint through a temporary directory;
- verifies every file against `checkpoint_manifest.json`;
- atomically publishes only verified local checkpoints;
- downloads final model, evaluation, logs, provenance, and manifests;
- terminates the pod only after the final-artifact gate passes.

## Individual controls

```powershell
& $python -m training.runpod_agent.status --run-document $run
& $python -m training.runpod_agent.download_checkpoints --run-document $run
& $python -m training.runpod_agent.download_final --run-document $run
& $python -m training.runpod_agent.terminate --run-document $run
```

Termination refuses to proceed until final artifacts are verified. The
`--force` flag exists only for an intentionally abandoned or failed run and
must be used explicitly.

The combined CLI is also available:

```powershell
& $python -m training.runpod_agent --help
```

## Recovery contract

A controller crash does not imply a training restart. A replacement agent reads
`run.json`, checks the provider and remote PID, calls `start-training` to
reattach if necessary, then restarts `watch`. The trainer discovers the latest
checkpoint only when no active process exists. Existing remote and local
checkpoints are never deleted by launch or recovery commands.
