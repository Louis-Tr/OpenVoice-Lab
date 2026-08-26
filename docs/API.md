# API contract

Stage 4 extends the synthesis response with measurements from the same inference
operation that produced the returned audio. The browser displays this contract
without knowing how the backend runtime collects it. Generated WAV files are
served locally; no external inference API is used.

## `POST /api/synthesis`

Create an audio synthesis result from text and a technology-neutral model/voice
selection.

Request:

```json
{
  "text": "OpenVoice Lab measures local inference.",
  "modelId": "kokoro",
  "voiceId": "af_heart",
  "variant": "fp32"
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

List selectable models, voices, deployed variants, and public runtime metadata
without exposing local artifact paths.

Current response:

```json
[
  {
    "id": "kokoro",
    "displayName": "Kokoro",
    "voices": ["af_heart"],
    "variants": ["fp32"],
    "modelVersion": "1.0",
    "runtime": "ONNX",
    "hosting": "self-hosted",
    "externalInferenceApis": [],
    "available": true
  }
]
```

Current behavior: `200 OK` with public model/runtime metadata and local artifact
availability. Listing metadata does not load the ONNX session.

TODO: define richer capabilities, readiness policy, pagination, and stable
identifier policy.

## `POST /api/benchmarks`

Trigger the predefined sentence corpus for a selected model variant and return
or schedule aggregate results.

Placeholder request:

```json
{
  "model_id": "model-id",
  "variant": "quantized"
}
```

Response:

```json
{
  "benchmark_id": "benchmark-id",
  "status": "pending",
  "aggregates": {}
}
```

Current scaffold behavior: `501 Not Implemented`.

TODO: choose synchronous versus job-based execution, define progress and
cancellation, version the sentence corpus, and specify aggregate statistics.

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
