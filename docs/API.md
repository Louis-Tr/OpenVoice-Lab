# API contract

The Kokoro product and SpeechT5 experiment use separate API namespaces and
registries. Both reuse backend-owned text processing, keep generated WAV files
local, and use no external inference or text-processing API. Historical Stage
11 evidence and live Stage 12 CPU measurements are distinct response fields.

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

## Stage 12 experiment API

### `GET /api/experiments/stage11/report`

Return a verified projection of the completed Stage 11 artifacts: integrity
state, frozen configuration, source audio/leakage audit, per-variant exposure
strategies and manifest hashes, exact validation histories, shared-test
evaluation, checkpoint inventory, selected-model hashes, GPU time, cost,
resumptions, incidents, and the zero-step pretrained control evaluated on the
same locked 662-case manifest. The endpoint fails closed with `503` if the
dataset lock, final audit, revisions, model files, pretrained evidence, or
artifact hashes disagree.

### `GET /api/experiments/stage11/fixtures`

List text-only rows from the locked shared test manifest. Query parameters are
`query`, `term`, `category`, `offset` (default `0`), and `limit` (default `30`,
maximum `100`). The response includes the locked manifest SHA-256. Audio source
paths are never exposed to the browser.

### `GET /api/experiments/stage11/models`

Return the immutable live-comparison catalog for pretrained SpeechT5 and the
three selected Stage 11 variants. Public metadata includes ID, role, revision,
model SHA-256, CPU runtime label, self-hosted state, and availability; local
paths are private.

### `POST /api/experiments/stage11/comparisons`

Start a durable, concurrency-limited CPU comparison. Fixture mode resolves text
and terms from the locked manifest:

```json
{
  "mode": "fixture",
  "fixtureId": "medical-3ac812069e511fd83561",
  "modelIds": ["speecht5-pretrained", "speecht5-v3-replay"],
  "sanitizeText": true,
  "normalizeText": true
}
```

Custom mode requires every explicit target term to appear in the submitted
text:

```json
{
  "mode": "custom",
  "text": "The patient was prescribed amlodipine for hypertension.",
  "targetTerms": ["amlodipine", "hypertension"],
  "modelIds": ["speecht5-pretrained", "speecht5-v3-replay"],
  "sanitizeText": true,
  "normalizeText": false
}
```

At least two and at most four unique model IDs are required. Both text options
default to `true` and remain independent. A successful start returns `202` with
the durable job snapshot. Queue saturation returns `429`; missing optional CPU
dependencies or pinned artifacts return `503`; malformed input returns `422`.

Each successful per-model result contains the exact original/final text,
progressive audio URL, local ASR transcript, correct/missed target terms, term
accuracy, WER, CPU load/inference/ASR/duration/RTF/RSS/warm metrics, model hash,
source revision, vocoder revision, and speaker-profile hash. Per-model failures
remain in the final comparison rather than discarding successful outputs.

### `GET /api/experiments/stage11/comparisons/{jobId}`

Return the latest durable snapshot. Terminal stages are `completed`,
`completed_with_failures`, `failed`, and `cancelled`.

### `GET /api/experiments/stage11/comparisons/{jobId}/events`

Stream the same snapshots as server-sent events. Angular falls back to polling
if SSE disconnects; the contract is identical.

### `DELETE /api/experiments/stage11/comparisons/{jobId}`

Request cooperative cancellation. A currently executing CPU operation is not
corrupted; the terminal cancellation is preserved atomically with its manifest.
Completed jobs are returned unchanged.

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
