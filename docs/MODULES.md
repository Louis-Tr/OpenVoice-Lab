# Module responsibilities

The tables below are ownership rules. A module may call another module through
its public contract, but it should not absorb the other module's responsibility.

## Frontend

| Module | Responsibility |
| --- | --- |
| `synthesis` | Text → voice/model selection → synthesis request workflow. |
| `audio-player` | Playback and audio duration. |
| `metrics` | Display inference latency, RTF, memory and model metadata. |
| `benchmark` | Trigger benchmark runs and display aggregate results. |
| `model-selector` | Retrieve/select available voices and FP32/quantized variants. |
| `api` | Angular HTTP abstraction around backend APIs. |
| `shared` | Common components, types and UI states. |
| `core` | Application configuration, API base URL and global concerns. |

## Backend

| Module | Responsibility |
| --- | --- |
| `api` | Thin HTTP controllers only. |
| `schemas` | Pydantic API contracts. |
| `synthesis` | Orchestration of the complete synthesis workflow. |
| `models` | Model registry, model loading and lifecycle. |
| `inference` | Abstract TTS inference interface plus Kokoro ONNX implementation. |
| `audio` | Encoding, duration and audio-file handling. |
| `metrics` | Latency, process memory, RTF and cold/warm metrics. |
| `benchmark` | Predefined test execution and result aggregation. |
| `config` | Environment/model/deployment configuration. |
| `health` | Liveness/readiness support. |

## Stage 7 implementation status

The connected vertical slice is intentionally narrow:

- Implemented frontend path: `synthesis` → `model-selector` → `api` →
  `audio-player` + `metrics`.
- Implemented backend path: `api` → `synthesis` → `models` → `inference` →
  `metrics` → `audio`.
- Implemented variants: API-discovered `kokoro-fp32` and `kokoro-q8`
  configurations, each with independent lifecycle and measurements.
- Implemented benchmark CLI: hashed corpus → isolated model workers →
  `SynthesisService` → raw outcomes → aggregates → timestamped JSON.
- Implemented benchmark product path: Angular trigger → FastAPI job service →
  progress polling → failure recovery → comparison table.
- Still deferred: durable/distributed job storage and advanced analytics.

## Boundary enforcement

- Technology names such as Kokoro and ONNX stay below the backend inference
  abstraction.
- HTTP serialization and status-code concerns stay in `api`.
- Cross-collaborator workflow order stays in `synthesis`.
- Feature-specific frontend state stays in its owning feature; only genuinely
  reusable pieces move to `shared` or `core`.
