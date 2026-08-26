# API contract

Stage 6 adds an executable CLI benchmark over the same synthesis service used by
the API. The benchmark HTTP endpoint remains an explicit job-orchestration
placeholder; no synchronous success response is claimed. Generated WAV files
are served locally, and no external inference API is used.

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

Planned asynchronous transport for the benchmark system. The current complete
execution path is the CLI command documented below.

Placeholder request:

```json
{
  "modelIds": ["kokoro-fp32", "kokoro-q8"],
  "voiceId": "af_heart"
}
```

Current response:

```json
{
  "detail": "Benchmark HTTP job orchestration is not implemented yet."
}
```

Current HTTP behavior: `501 Not Implemented`. This is not the Stage 6 execution
surface and is not presented as working.

The executable benchmark is:

```powershell
cd backend
.\.venv\Scripts\python.exe -m app.benchmark.runner
```

It writes a timestamped JSON file containing corpus version/hash, environment,
raw case outcomes, and per-model aggregates to `backend/benchmark-results/`.

TODO: define job creation, progress, cancellation, result retention, and result
retrieval for the HTTP transport.

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
