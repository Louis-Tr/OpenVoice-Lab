# API contract

Stage 10 exposes two independent preprocessing options. Deterministic English
normalization runs before sanitization, the original request remains visible,
and `normalizedText` records the exact string passed to inference. Generated
WAV files remain local; no external inference or text-processing API is used.

## `POST /api/synthesis`

Create an audio synthesis result from text and a technology-neutral model/voice
selection.

Request:

```json
{
  "text": "Save 15% at [https://example.com](https://example.com). Price: $25.",
  "modelId": "kokoro-fp32",
  "voiceId": "af_heart",
  "sanitizeText": true,
  "normalizeText": true
}
```

Genuine warm response recorded on the local development machine:

```json
{
  "status": "ok",
  "model": "kokoro-fp32",
  "text": "Save 15% at [https://example.com](https://example.com). Price: $25.",
  "normalizedText": "Save 15 percent at example dot com. Price: 25 dollars.",
  "audioUrl": "/audio/kokoro-fp32-af_heart-7fd22f34e5a3c2e0.wav",
  "metrics": {
    "modelLoadMs": 0.0,
    "inferenceMs": 895.835,
    "audioDurationMs": 4853.333,
    "realTimeFactor": 0.184581,
    "memoryMb": 878.172,
    "warm": true,
    "modelVariant": "fp32"
  }
}
```

The returned 233,004-byte artifact was retrieved successfully. Metric values
are evidence for this machine and request, not fixed service guarantees.

Current behavior: `normalizeText` and `sanitizeText` both default to `true` and
operate independently. Normalization expands supported English notation;
sanitization then performs Unicode, whitespace, control-character, and
punctuation-noise cleanup. With both disabled, the validated request string is
sent unchanged. `normalizedText` always equals the exact inference input while
`text` remains the original request value. Processing happens before model
loading and is excluded from `inferenceMs`. If enabled sanitization leaves no
alphanumeric content, the service returns `422`. Missing or malformed values
also receive Pydantic `422`; unsupported voices return `422`, unknown models
return `404`, unavailable artifacts return `503`, and inference, measurement,
or storage failures return `500`.

Supported deterministic transformations include USD `$` amounts, numeric
percentages, email addresses, HTTP(S) URLs, relative `./` paths, Markdown links
and emphasis, inline code, common comparison operators, snake case, and camel
case. The implementation is English-focused and deliberately does not attempt
nested Markdown, arbitrary source-code parsing, non-USD currencies, or general
natural-language number expansion.

| `sanitizeText` | `normalizeText` | Processing |
| --- | --- | --- |
| `true` | `true` | Normalize, then sanitize. |
| `true` | `false` | Sanitize raw input only. |
| `false` | `true` | Normalize supported notation only. |
| `false` | `false` | Preserve validated input exactly. |

Measurement semantics:

- `modelLoadMs`: cold model-loader boundary; `0` when the runtime is reused.
- `inferenceMs`: wall-clock engine execution, excluding loading and WAV storage.
- `audioDurationMs`: exact sample count divided by sample rate.
- `realTimeFactor`: `inferenceMs / audioDurationMs`, rounded to six decimals.
- `memoryMb`: process resident set size after inference.
- `warm`: whether the engine was already loaded before this request.
- `modelVariant`: the measured deployed precision variant.

TODO: define artifact retention and cancellation.

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
  "voiceId": "af_heart",
  "sanitizeText": true,
  "normalizeText": true
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

Current behavior: validates model IDs, voice, and independent preprocessing
flags, creates an in-memory job, and returns before inference begins. Both flags
default to `true`. The background job runs one isolated process per model and
persists the merged JSON result. Each raw result records original `text`, final
`normalizedText`, `sanitizeText`, and `normalizeText`; the run-level result
records the same processing configuration. Failed cases preserve exact final
text when preprocessing completed and use `null` when no honest final value
exists.

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
