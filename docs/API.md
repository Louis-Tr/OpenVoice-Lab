# API placeholders

This document reserves the initial HTTP surface. Field names reflect the
scaffolded Pydantic and TypeScript contracts, but compatibility guarantees,
error taxonomy, artifact delivery, and asynchronous execution semantics remain
to be defined before feature implementation.

## `POST /api/synthesis`

Create an audio synthesis result from text and a technology-neutral model/voice
selection.

Placeholder request:

```json
{
  "text": "Text to synthesize",
  "model_id": "model-id",
  "voice_id": "voice-id",
  "variant": "fp32"
}
```

Placeholder success response:

```json
{
  "audio_url": "/artifacts/audio/example.wav",
  "duration_seconds": 0,
  "metrics": {
    "latency_ms": 0,
    "real_time_factor": 0,
    "memory_mb": 0,
    "cold_start": true,
    "model_id": "model-id",
    "variant": "fp32"
  }
}
```

Current scaffold behavior: `501 Not Implemented`.

TODO: define artifact lifetime, error responses, request limits, cancellation,
and cold/warm classification.

## `GET /api/models`

List selectable models, voices, and FP32/quantized variants without exposing
runtime or artifact implementation details.

Placeholder response:

```json
[
  {
    "id": "model-id",
    "display_name": "Model name",
    "voices": ["voice-id"],
    "variants": ["fp32", "quantized"]
  }
]
```

Current scaffold behavior: `200 OK` with an empty list.

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

Placeholder response:

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
  "status": "ok"
}
```

Current scaffold behavior: `200 OK` when the FastAPI process is live.

TODO: decide whether readiness receives a separate endpoint and which model,
storage, and runtime dependencies gate it.

