# Google Cloud Run deployment

OpenVoice Lab ships a root `Dockerfile` for Google Cloud source builds. It
produces one container: Angular is compiled during the build, FastAPI serves the
SPA and API on Cloud Run's injected `PORT`, and the three Kokoro ONNX variants
plus pretrained SpeechT5 CPU are downloaded and SHA-256
verified before entering the final image.

## Continuous deployment settings

In the Cloud Run **Set up with Cloud Build** flow, use:

| Setting | Value |
| --- | --- |
| Branch | `main` |
| Build type | Dockerfile |
| Source location | `/Dockerfile` |
| Authentication | Allow public access for the portfolio site |

`/Dockerfile` is correct only after the new root file has been committed and
pushed to `main`. Its directory is the build context, so the root location gives
the build access to both `frontend/` and `backend/`.

Use this initial Cloud Run service configuration:

| Setting | Recommended value | Reason |
| --- | --- | --- |
| CPU | 4 vCPU | Allows two measured two-core inference commitments when memory also fits. |
| Memory | 8 GiB | Leaves explicit headroom for cached engines, overlapping work, and audio buffers. |
| Request timeout | 600 seconds | Allows long synthesis requests without using the 60-minute platform maximum. |
| Billing | Instance-based | Benchmark work continues outside the request that starts a job. |
| Minimum instances | 0 | Avoids paying continuously when the portfolio is idle. |
| Maximum instances | 1 | Keeps in-process jobs and ephemeral audio on one instance. |
| Concurrency | 8 | Lets enqueue, polling, audio, and health requests proceed while the in-process scheduler bounds compute. |

The application health endpoint is `/health`. The container runs as an
unprivileged user and writes generated audio, benchmark output, and job state
only below `/tmp/openvoice`.

## What the cloud image exposes

- Kokoro FP32, FP16, and INT8 are available for live synthesis.
- The pretrained SpeechT5 control runs on CPU with a pinned model, vocoder, and
  CMU speaker embedding.
- The SpeechT5 experiment tab serves a committed, SHA-256-verified snapshot of
  the measured training report and its 350 fixtures.
- The pinned Whisper evaluator is packaged for live experiment scoring.
- Adapted SpeechT5 models are downloaded only when selected. Every file is
  checked against the committed remote-model inventory before the runtime can
  load it.

The product model loader retains up to two idle engines in this 8 GiB profile.
Every synthesis, benchmark, experiment inference, and experiment ASR operation
must first acquire an in-process resource reservation. HTTP concurrency is set
above one so job submission and status polling remain responsive; it does not
set model concurrency. Keep one Uvicorn worker and a maximum of one Cloud Run
instance while queue state and generated audio remain instance-local.

The default queue backfills fitting requests for up to 30 seconds. At that
boundary, the oldest request reserves the next compute start and later work no
longer bypasses it. Cloud Run should use instance-based billing because accepted
asynchronous work continues after the submit response returns.

If no remote model origin is configured, the snapshot remains available and the
UI explains that live comparisons are not provisioned. It does not imply that a
disabled model is runnable.

## Provision live Stage 11 comparisons

The Git repository intentionally contains model identities and hashes, not the
adapted weights. Prepare an upload tree from the four verified selected models:

```powershell
python backend/scripts/export_experiment_remote_models.py --stage-files
```

Upload the contents of `artifacts/stage12/remote-models/models/` to private
object storage without changing its directory layout:

```text
<origin>/
  speecht5-v1a-conservative-full/model.safetensors
  speecht5-v1b-lora/model.safetensors
  speecht5-v1c-gradual-unfreeze/model.safetensors
  speecht5-v1d-reduction-factor-1/model.safetensors
  ...the remaining files listed for each model...
```

Configure these Cloud Run environment variables:

| Variable | Value |
| --- | --- |
| `OPENVOICE_EXPERIMENT_MODEL_BASE_URL` | HTTPS root containing the four model directories above |
| `OPENVOICE_EXPERIMENT_MODEL_ACCESS_TOKEN` | Optional bearer token for a private origin |

Store the access token in Secret Manager and expose it to Cloud Run as a secret;
never put it in source, image layers, build arguments, or ordinary environment
configuration committed to Git. A public origin can omit the token.

At runtime the service downloads into a temporary directory, verifies byte
counts and SHA-256 digests from `remote_models.json`, writes a completion
marker, and atomically promotes the model into the instance-local cache. An
interrupted or corrupt download is never presented as an available local model.

## Persistence limits

Cloud Run's writable filesystem is ephemeral and instance-local. Generated WAVs,
benchmark results, and background-job state can disappear when the instance is
replaced or scaled to zero. Maximum instance count `1` makes the current
portfolio deployment coherent, but it does not make those files durable.

Before operating this as a multi-instance or durable service, move audio and
result objects to Cloud Storage and job coordination to a persistent service.
Do not raise the maximum instance count until that migration is complete.

## Local production-image verification

Build and exercise the same entry point used by Cloud Run:

```powershell
docker build --tag openvoice-lab:cloud-run --file Dockerfile .
docker run --rm --publish 8080:8080 --env PORT=8080 openvoice-lab:cloud-run
```

Then open `http://localhost:8080` or check:

```powershell
curl http://localhost:8080/health
```

The source-build workflow and Dockerfile-context rules are documented by
[Cloud Run continuous deployment](https://docs.cloud.google.com/run/docs/continuous-deployment)
and [Cloud Run source builds](https://docs.cloud.google.com/run/docs/building/containers).
Resource and background-processing choices follow Cloud Run's
[memory](https://docs.cloud.google.com/run/docs/configuring/services/memory-limits),
[timeout](https://docs.cloud.google.com/run/docs/configuring/request-timeout),
and [billing](https://docs.cloud.google.com/run/docs/configuring/billing-settings)
documentation.
