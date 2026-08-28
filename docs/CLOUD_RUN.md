# Google Cloud Run deployment

OpenVoice Lab ships a root `Dockerfile` for Google Cloud source builds. It
produces one container: Angular is compiled during the build, FastAPI serves the
SPA and API on Cloud Run's injected `PORT`, and the three Kokoro ONNX variants
plus Audio8 INT4 ONNX and pretrained SpeechT5 CPU are downloaded and SHA-256
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
| CPU | 2 vCPU | CPU-hosted ONNX inference benefits from parallel execution. |
| Memory | 4 GiB | Leaves headroom for the service, one bounded model session, and generated audio. |
| Request timeout | 600 seconds | Allows long synthesis requests without using the 60-minute platform maximum. |
| Billing | Instance-based | Benchmark work continues outside the request that starts a job. |
| Minimum instances | 0 | Avoids paying continuously when the portfolio is idle. |
| Maximum instances | 1 | Keeps in-process jobs and ephemeral audio on one instance. |
| Concurrency | 1 | Prevents simultaneous model loads or synthesis calls from multiplying memory pressure. |

The application health endpoint is `/health`. The container runs as an
unprivileged user and writes generated audio, benchmark output, and job state
only below `/tmp/openvoice`.

## What the cloud image exposes

- Kokoro FP32, FP16, and INT8 are available for live synthesis.
- Audio8 uses the official INT4 ONNX export on CPU with the API voice
  `unconditioned`; its optional voice-registration model is not packaged.
- The pretrained SpeechT5 control runs on CPU with a pinned model, vocoder, and
  CMU speaker embedding.
- The SpeechT5 experiment tab serves a committed, SHA-256-verified snapshot of
  the measured training report and its 350 fixtures.
- Live adapted-model comparisons remain unavailable because the selected
  multi-gigabyte Stage 11 checkpoints and ASR evaluator are not packaged.

The product model loader retains only one engine in this 4 GiB deployment.
Switching models may therefore incur a cold load, but avoids retaining Audio8,
SpeechT5, and Kokoro sessions together. Keep concurrency at `1`; these CPU
variants are portfolio-scale interactive paths, not high-throughput serving.

The snapshot preserves measured evidence; it does not manufacture live model
availability. The API returns an explicit capability error for unsupported live
experiment runs.

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
