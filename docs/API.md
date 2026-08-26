# API contract

Stage 7 adds an asynchronous browser contract over the Stage 6 benchmark. The
POST endpoint returns immediately, Angular polls job progress, and the completed
response contains the same raw and aggregate evidence as the CLI. Generated WAV
files are served locally, and no external inference API is used.

## `POST /api/synthesis`

Create an audio synthesis result from text and a technology-neutral model/voice
selection.

Request:

```json
{
  "text": "OpenVoice Lab measures local inference.",
  "modelId": "kokoro-fp32",
  "voiceId": "af_heart"
}
```

Current success response:

```json
{
  "status": "ok",
  "model": "kokoro-fp32",
  "text": "OpenVoice Lab measures local inference.",
  "audioUrl": "/audio/kokoro-fp32-af_heart-5cf95926dd1375d6.wav",
  "metrics": {
    "modelLoadMs": 0.0,
    "inferenceMs": 676.107,
    "audioDurationMs": 2782.667,
    "realTimeFactor": 0.242971,
    "memoryMb": 529.816,
    "warm": true,
    "modelVariant": "fp32"
  }
}
```

Current behavior: `200 OK` with a playable local WAV for valid input. Missing
fields and malformed values receive Pydantic `422` responses. Unsupported voices
return `422`, unknown models return `404`, unavailable artifacts return `503`,
and inference/measurement/storage failures return `500`.

Measurement semantics:

- `modelLoadMs`: cold model-loader boundary; `0` when the runtime is reused.
- `inferenceMs`: wall-clock engine execution, excluding loading and WAV storage.
- `audioDurationMs`: exact sample count divided by sample rate.
- `realTimeFactor`: `inferenceMs / audioDurationMs`, rounded to six decimals.
- `memoryMb`: process resident set size after inference.
- `warm`: whether the engine was already loaded before this request.
- `modelVariant`: the measured deployed precision variant.

TODO: define artifact retention, request limits, and cancellation.

## `GET /api/models`

List selectable deployable configurations, voices, precision, and public runtime
metadata without exposing local artifact paths.

Current response:

```json
[
  {
    "id": "kokoro-fp32",
    "name": "Kokoro",
    "precision": "FP32",
    "variant": "fp32",
    "voices": ["af_heart"],
    "modelVersion": "1.0",
    "runtime": "ONNX",
    "hosting": "self-hosted",
    "externalInferenceApis": [],
    "available": true
  },
  {
    "id": "kokoro-q8",
    "name": "Kokoro",
    "precision": "INT8",
    "variant": "quantized",
    "voices": ["af_heart"],
    "modelVersion": "1.0",
    "runtime": "ONNX",
    "hosting": "self-hosted",
    "externalInferenceApis": [],
    "available": true
  }
]
```

Current behavior: `200 OK` with one entry per registry configuration and local
artifact availability. Listing metadata does not load either ONNX session.

TODO: define richer capabilities, readiness policy, and pagination.

## `POST /api/benchmarks`

Start the complete fixed-corpus benchmark in background model workers.

Request:

```json
{
  "modelIds": ["kokoro-fp32", "kokoro-q8"],
  "voiceId": "af_heart"
}
```

Current `202 Accepted` response:

```json
{
  "benchmarkId": "benchmark-2026-08-26T11-18-24-802081Z",
  "status": "pending",
  "testCaseCount": 8,
  "modelCount": 2,
  "totalEvaluations": 16,
  "completedEvaluations": 0,
  "progressPercent": 0.0,
  "result": null,
  "error": null
}
```

Current behavior: validates model IDs and voice, creates an in-memory job, and
returns before inference begins. The background job runs one isolated process
per model and persists the merged Stage 6 JSON result.

## `GET /api/benchmarks/config`

Describe the fixed browser workload without loading a model.

```json
{
  "corpusVersion": "1.0.0",
  "corpusSha256": "eaf6215e4cf13e670e0b3cfb56f33b6a50939a61e30a2b3296ed7c44d1d9cb98",
  "testCaseCount": 8,
  "modelCount": 2,
  "totalEvaluations": 16,
  "modelIds": ["kokoro-fp32", "kokoro-q8"],
  "defaultVoiceId": "af_heart"
}
```

## `GET /api/benchmarks/{benchmarkId}`

Poll one job. `pending` and `running` responses expose completed and total
evaluation counts. `completed` includes the full `BenchmarkResult`; `failed`
includes an error and no fabricated result.

## `GET /api/benchmarks/latest`

Return the newest in-memory job so navigation or a fresh browser session can
recover progress or results. Returns `404` before any job has been started or
after a backend process restart.

The executable benchmark is:

```powershell
cd backend
.\.venv\Scripts\python.exe -m app.benchmark.runner
```

It writes a timestamped JSON file containing corpus version/hash, environment,
raw case outcomes, and per-model aggregates to `backend/benchmark-results/`.

TODO: define cancellation, durable job storage, multi-process coordination,
retention, and paginated raw-result retrieval for production.

## `GET /health`

Report service liveness and, later, dependency readiness.

Response:

```json
{
  "status": "healthy"
}
```

Current behavior: `200 OK` when the FastAPI process is live.

TODO: decide whether readiness receives a separate endpoint and which model,
storage, and runtime dependencies gate it.
