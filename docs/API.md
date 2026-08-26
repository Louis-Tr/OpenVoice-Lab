# API contract

Stage 3 keeps the established API shape and adds its Angular consumer. The
browser discovers models through `GET /api/models`, submits the same synthesis
contract, and plays the returned `/audio` URL. Generated WAV files are served
locally; no external inference API is used.

## `POST /api/synthesis`

Create an audio synthesis result from text and a technology-neutral model/voice
selection.

Request:

```json
{
  "text": "OpenVoice Lab is running locally.",
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
  "text": "OpenVoice Lab is running locally.",
  "audioUrl": "/audio/kokoro-fp32-af_heart-27561063304ff41f.wav"
}
```

Current behavior: `200 OK` with a playable local WAV for valid input. Missing
fields and malformed values receive Pydantic `422` responses. Unsupported voices
return `422`, unknown models return `404`, unavailable artifacts return `503`,
and inference/storage failures return `500`.

TODO: define artifact retention, request limits, cancellation, and cold/warm
metrics.

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
