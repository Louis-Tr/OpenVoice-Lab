# API contract

The product synthesis catalog and SpeechT5 experiment use separate API
namespaces and registries. The product catalog can expose a pinned pretrained
SpeechT5 serving configuration, but it does not expose experiment reports,
adapted checkpoints, or Whisper scoring. Both paths reuse backend-owned text
processing, keep generated WAV files local, and use no external inference or
text-processing API.

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

Product SpeechT5 accepts up to 5,000 input characters, like Kokoro. The existing
5,000-character post-cleanup limit also applies. Its `maxInputTokens: 600` is a
**per-chunk** model limit, not a limit on the complete request. Longer processed
text is split without truncation, preferring sentence boundaries, then whitespace,
then character boundaries for overlong words. Every chunk is checked with the
model tokenizer, including special tokens. Inputs already within the token budget
remain one unchanged generation call.

Chunks run sequentially on the same engine and speaker profile, with one
request-local seed-42 generator and one scheduler lease through final audio storage.
Each chunk is vocoded separately; the 16 kHz waveforms are concatenated in order
without extra silence or crossfading. Chunk boundaries may change prosody. Timing
and RSS instrumentation cover the complete engine call, including splitting and
concatenation. A failed chunk fails the whole request; no partial audio is saved.
Job status remains `generating` across chunks. This does not change the separate
experiment-comparison runtime. The 1,600 MiB admission estimate is not a hard memory
cap: maximum-length, warm and concurrent runs still require capacity measurement.

A local 5,000-character smoke check on Windows with two CPU threads completed
10 chunks (largest: 571 tokens), producing 270.624 seconds of finite, non-silent
16 kHz mono audio in 294.807 seconds. Sampled process RSS peaked at 3,980.055 MiB.
This was one cold run, not an 8 GB-constrained or concurrent capacity test, and
does not establish worst-case memory or perceptual quality. In particular, it
exceeds the existing 1,600 MiB SpeechT5 admission estimate. That fixed estimate
has not been changed automatically and must be retuned before relying on admission
to protect concurrent long-form workloads. Repeated short-input generation also
matched within numerical tolerance after this run.

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

## Synthesis jobs and idempotency

`POST /api/synthesis/jobs` accepts the same JSON body as `POST /api/synthesis`,
plus a required `Idempotency-Key` header (1–128 ASCII letters, digits, `.`, `_`,
`:`, or `-`). Use a fresh random UUID for each deliberate generation. The
existing synchronous endpoint is unchanged and does not provide idempotency.

```powershell
$body = @{ text = 'Hello'; modelId = 'kokoro-fp32'; voiceId = 'af_heart' } | ConvertTo-Json
$headers = @{ 'Idempotency-Key' = [guid]::NewGuid().ToString() }
$job = Invoke-RestMethod -Method Post -Uri 'http://localhost:8000/api/synthesis/jobs' `
    -Headers $headers -ContentType 'application/json' -Body $body
Invoke-RestMethod "http://localhost:8000/api/synthesis/jobs/$($job.jobId)"
# Repeating the POST with the SAME headers and body observes this same job.
```

The service atomically claims a key before starting work. All validated request
fields participate in the fingerprint; JSON property order and omitted defaults
do not change it. Identical keys with different input receive `409` with the
existing `{"detail": ...}` error format. Different keys can generate identical
input independently; this is job idempotency, not an audio result cache.
Replays never acquire another engine lease, including during cold construction.

Submission waits for scheduler resource admission, but not model construction,
inference, or WAV creation. Insufficient CPU/memory, pending model loads, or a
full job store return `503` before acceptance. Unknown models return `404` and
invalid input/voices return `422`. Resource/input rejection after key claim is
retained as `rejected`: the same key replays the same HTTP error. Use a new key
for a deliberate retry after a definitive rejection. Header/Pydantic validation
and store-capacity rejections occur before key claim and do not consume a key.

Accepted, unfinished jobs return `202`; already terminal accepted jobs return
`200`, including a fast first execution that finishes before the response.
The POST supplies `Location` for status polling and `Retry-After: 2` while active.
Both success endpoints use `Cache-Control: no-store`.

`GET /api/synthesis/jobs/{jobId}` returns `200` with the current snapshot:

```json
{
  "jobId": "f8b131a9069946758006810c4e6b3c46",
  "status": "generating",
  "request": {
    "text": "Hello", "modelId": "kokoro-fp32", "voiceId": "af_heart",
    "sanitizeText": true, "normalizeText": true
  },
  "createdAt": "2026-09-07T12:00:00Z",
  "updatedAt": "2026-09-07T12:00:01Z",
  "completedAt": null,
  "expiresAt": null,
  "result": null,
  "error": null
}
```

Stages are `pending` (validation/admission), `loading`, `generating`, `saving`,
and terminal `completed`, `failed`, or `rejected`. A completed job embeds the
unchanged `SynthesisResult`, including audio URL and original measurements.
Failures embed `error: {"statusCode": 500, "detail": "..."}` (the code reflects
the error category). GET remains `200` for an existing failed job; missing or
expired jobs return `404`. Retrying a failed accepted job returns its recorded
failure without running inference again. Stage timestamps are observations,
not exact engine progress percentages; existing RSS/timing logs remain intact.

To reconnect after a browser refresh, a client must save the key and exact
request **before** POST, then save the returned job ID. On refresh, GET that job
and resume polling. If the initial response was lost, repeat POST with the saved
key/body. A transient polling/network error does not mean generation failed.
The Angular synthesis page uses this flow. It saves the latest submitted text,
model, voice, cleanup flags, and key in `localStorage` under
`openvoice.synthesis.v1:<apiBaseUrl>`. After acknowledgement it also saves the job
ID. Reloading or returning to the synthesis page restores these fields and polls
the known job immediately, including restoring completed audio and measurements.
Model-catalog loading does not replace the restored selection. Unsaved edits
before Generate are not persisted; the saved record represents a submission.

The page polls every two seconds while active, without overlapping slow GETs,
and stops on a terminal status or navigation. A network error preserves the
saved identity and offers **Check generation**; it never enables a second
submission while the first outcome is unknown. A confirmed rejection or missing
job restores editable input without automatically creating replacement work.
An explicit Generate after completion/rejection uses a new key. Unacknowledged
submissions older than 24 hours require an explicit new generation rather than
being replayed automatically. Jobs with known IDs are always recovered via GET.

Browser storage must be writable before a new POST is sent. Malformed saved
records are ignored, and blocked storage is reported on the page. The latest
submitted text remains on this browser until replaced or site storage is
cleared; audio and metrics are fetched from the backend, not cached in browser
storage. Backend restart/expiry still prevents restoring a removed job/result.

The job service owns background tasks independently of the HTTP request.
Disconnects do not cancel generation. Graceful shutdown drains jobs; the engine
lease remains held through WAV creation and is released on worker completion or
failure. There is no job cancellation API, automatic retry, or waiting resource
queue. Admission, replay, and terminal transitions emit structured logs with job
ID, model ID, and status, without request text or idempotency keys.

The initial `JobStore` is in memory, scoped to this synthesis endpoint in one
serving process. Keys are not authentication: this public application has no
user accounts, and possession of a job ID grants access to its snapshot. Use
unguessable keys/IDs; add authenticated owner scoping if accounts are introduced.
There is intentionally no global "latest job" recovery endpoint.

`OPENVOICE_SYNTHESIS_JOB_MAXIMUM_RECORDS` defaults to 1000;
`OPENVOICE_SYNTHESIS_JOB_RETENTION_SECONDS` defaults to 86400. Terminal records
and their keys expire together after this interval, with lazy cleanup on
GET/POST. Active jobs never expire. A full store rejects new identities rather
than discarding promised idempotency records. After expiry, a key may create a
new generation. This does not delete audio files: existing shared artifact
storage/retention remains unchanged.

Backend restart loses identities and status. Run one serving process/instance;
multiple workers would have independent stores and scheduler budgets. Cloud Run
instance replacement also loses local audio. Durable or multi-instance recovery
requires shared job coordination and audio storage; this implementation does
not claim exactly-once execution across process crashes.

## `GET /api/models`

List the full serving catalog, including capability state, voices, precision,
and public runtime metadata without exposing local artifact paths. Angular
disables entries whose `available` value is `false` and presents
`unavailableReason`; it does not infer readiness from model names.

Current catalog:

| ID | Display | Precision | Runtime | Current state |
| --- | --- | --- | --- | --- |
| `kokoro-fp32` | Kokoro | FP32 | ONNX Runtime | Ready with verified artifact |
| `kokoro-fp16` | Kokoro | FP16 | ONNX Runtime | Ready with verified artifact |
| `kokoro-q8` | Kokoro | INT8 | ONNX Runtime | Ready with verified artifact |
| `audio8-0.6b` | Audio8 0.6B | INT4 | ONNX Runtime CPU | Ready with pinned official export |
| `speecht5-pretrained` | SpeechT5 | FP32 | PyTorch CPU | Ready with pinned model, vocoder, and speaker profile |

Every item also returns `variant`, `voices`, `modelVersion`, `hosting`,
`externalInferenceApis`, `available`, `unavailableReason`, and `description`.
Listing metadata does not load an inference engine.

Audio8 exposes the fixed `unconditioned` voice; SpeechT5 exposes `cmu-slt`.
These are product-serving contracts. Audio8 voice cloning and adapted SpeechT5
experiment checkpoints are intentionally outside `POST /api/synthesis`.

TODO: define richer capabilities, readiness policy, and pagination.

## `POST /api/benchmarks`

Start the complete fixed-corpus benchmark in background model workers.

Request:

```json
{
  "modelIds": [
    "kokoro-fp32",
    "kokoro-fp16",
    "kokoro-q8",
    "audio8-0.6b",
    "speecht5-pretrained"
  ],
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
  "modelCount": 5,
  "totalEvaluations": 40,
  "completedEvaluations": 0,
  "progressPercent": 0.0,
  "result": null,
  "error": null
}
```

Current behavior: validates model IDs and independent preprocessing flags,
creates an in-memory job, and returns before inference begins. Omitting
`modelIds` selects every model currently available through `/api/models`. An
optional `voiceId` is honored when the selected model supports it; otherwise
each model uses its first advertised synthesis voice. Both preprocessing flags
default to `true`. The background job runs one isolated process per model and
persists the merged JSON result. Each raw result records original `text`, final
`normalizedText`, exact `voiceId`, `sanitizeText`, and `normalizeText`; the
run-level result records the same processing configuration and the resolved
`modelVoiceIds`. Failed cases preserve exact final text when preprocessing
completed and use `null` when no honest final value exists.

## `GET /api/benchmarks/config`

Describe the fixed browser workload without loading a model.

```json
{
  "corpusVersion": "1.0.0",
  "corpusSha256": "eaf6215e4cf13e670e0b3cfb56f33b6a50939a61e30a2b3296ed7c44d1d9cb98",
  "testCaseCount": 8,
  "modelCount": 5,
  "totalEvaluations": 40,
  "modelIds": [
    "kokoro-fp32",
    "kokoro-fp16",
    "kokoro-q8",
    "audio8-0.6b",
    "speecht5-pretrained"
  ],
  "modelVoiceIds": {
    "kokoro-fp32": "af_heart",
    "kokoro-fp16": "af_heart",
    "kokoro-q8": "af_heart",
    "audio8-0.6b": "unconditioned",
    "speecht5-pretrained": "cmu-slt"
  },
  "defaultVoiceId": null
}
```

The exact list is deployment-derived. Models missing required local artifacts
remain visible through `/api/models` as unavailable but are not included in a
benchmark run.

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
state, shared and approach-specific configuration, source audio/leakage audit,
the common V1 manifest hashes, exact validation histories, shared-test
evaluation, checkpoint inventory, selected-model hashes, GPU time, cost,
resumptions, incidents, and the zero-step pretrained control evaluated on the
same locked 662-case manifest. The endpoint fails closed with `503` if the
dataset lock, run completion records, revisions, model files, pretrained evidence, or
artifact hashes disagree.

### `GET /api/experiments/stage11/fixtures`

List text-only rows from the locked shared test manifest. Query parameters are
`query`, `term`, `category`, `offset` (default `0`), and `limit` (default `30`,
maximum `100`). The response includes the locked manifest SHA-256. Audio source
paths are never exposed to the browser.

### `GET /api/experiments/stage11/models`

Return the immutable live-comparison catalog for pretrained SpeechT5 and the
four selected V1 approach models. Public metadata includes ID, role, revision,
model SHA-256, CPU runtime label, self-hosted state, and availability; local
paths are private.

### `POST /api/experiments/stage11/comparisons`

Start a durable, concurrency-limited CPU comparison. Fixture mode resolves text
and terms from the locked manifest:

```json
{
  "mode": "fixture",
  "fixtureId": "medical-3ac812069e511fd83561",
  "modelIds": ["speecht5-pretrained", "speecht5-v1c-gradual-unfreeze"],
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
  "modelIds": ["speecht5-pretrained", "speecht5-v1b-lora"],
  "sanitizeText": true,
  "normalizeText": false
}
```

At least two and at most five unique model IDs are required. Both text options
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
