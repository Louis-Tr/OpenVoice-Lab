# API contract

Stage 1 makes health, model listing, and synthesis executable. Synthesis returns
deterministic mock data: it proves HTTP handling, Pydantic validation, and
service separation without claiming that inference works.

## `POST /api/synthesis`

Create an audio synthesis result from text and a technology-neutral model/voice
selection.

Request:

```json
{
  "text": "Text to synthesize",
  "modelId": "kokoro",
  "voiceId": "af_heart",
  "variant": "fp32"
}
```

Current success response:

```json
{
  "status": "mock",
  "model": "kokoro-fp32",
  "text": "Text to synthesize",
  "audioUrl": null
}
```

Current behavior: `200 OK` for valid input. Missing fields, empty text, invalid
variants, and other malformed inputs receive a Pydantic `422` response before
the service is called.

TODO: define artifact lifetime, error responses, request limits, cancellation,
and cold/warm classification.

## `GET /api/models`

List selectable models, voices, and FP32/quantized variants without exposing
runtime or artifact implementation details.

Current response:

```json
[
  {
    "id": "kokoro",
    "displayName": "Kokoro",
    "voices": ["af_heart"],
    "variants": ["fp32", "quantized"]
  }
]
```

Current behavior: `200 OK` with the configured Stage 1 catalog. This is metadata
only; it does not claim that a model artifact is loaded or ready.

TODO: define capability metadata, model availability/readiness, pagination,
and stable identifier policy.

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

Placeholder response:

```json
{
  "status": "healthy"
}
```

Current behavior: `200 OK` when the FastAPI process is live.

TODO: decide whether readiness receives a separate endpoint and which model,
storage, and runtime dependencies gate it.
