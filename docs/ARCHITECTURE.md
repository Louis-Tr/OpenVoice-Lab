# Architecture

## Purpose

OpenVoice Lab is split into an Angular presentation client and a FastAPI
application boundary. The repository begins with stable component boundaries so
that model runtimes and benchmarking mechanics can evolve without leaking into
the browser or HTTP controllers.

Stage 4 instruments the Stage 3 synthesis path without moving evaluation logic
into the inference adapter. A fresh browser request now receives its generated
WAV plus model-load, latency, duration, RTF, RSS memory, lifecycle, and variant
measurements. The repository does not yet execute aggregate benchmarks.

## System spine

```text
Angular
  ↓
REST
  ↓
FastAPI
  ↓
SynthesisService
  ↓
Inference abstraction
  ↓
Kokoro
  ↓
Audio
  ↓
Angular player
```

This dependency direction is executable end to end. ONNX and Kokoro remain
behind the backend inference abstraction; Angular sees only model, synthesis,
and audio URL contracts.

## Component boundaries

```text
┌──────────────────────────── Angular ─────────────────────────────┐
│ Feature components ──> API services ──> HTTP contracts only     │
└────────────────────────────────┬─────────────────────────────────┘
                                 │ /api
┌────────────────────────────────▼─────────────────────────────────┐
│ FastAPI controllers: validation, status codes, serialization    │
├────────────────────────────────┬─────────────────────────────────┤
│ SynthesisService: complete synthesis workflow orchestration     │
├───────────────┬────────────────┼────────────────┬────────────────┤
│ Model registry│ Inference port │ Metrics        │ Audio service  │
│ and lifecycle │ (abstract)     │ collector      │                │
├───────────────┴────────┬───────┴────────────────┴────────────────┤
│ Concrete adapters      │ Kokoro ONNX (initial adapter)           │
└────────────────────────┴─────────────────────────────────────────┘
```

### Angular boundary

Angular owns user interaction, UI state, playback, metric presentation, and
calls to documented backend contracts. It must not import, model, configure, or
otherwise depend on Kokoro, ONNX, runtime sessions, tensor formats, model-file
paths, or other inference implementation details.

The current synthesis page owns text and selection state, model discovery,
request lifecycle, and user-facing recovery messages. `model-selector` and
`audio-player` are presentational boundaries. `api` is the only frontend module
that performs HTTP calls. Local development proxies `/api` and `/audio` without
changing those public contracts.

### API boundary

FastAPI controllers are transport adapters. They may validate Pydantic request
contracts, invoke an application service, serialize its result, and map known
application errors to HTTP responses. They must not choose or load models, run
inference, encode audio, calculate RTF, inspect process memory, or aggregate
benchmark results.

The current executable path is:

```text
HTTP request
  ↓
FastAPI controller
  ↓
Pydantic validation
  ↓
Application service
  ↓
ModelRegistry
  ↓
ModelLoader (cached)
  ↓
MetricsCollector wraps TTSInferenceEngine
  ↓
KokoroONNXEngine
  ↓
AudioService
  ↓
Typed JSON response + local WAV
```

### Synthesis application boundary

`SynthesisService` owns the complete synthesis workflow. It currently
coordinates model resolution/lifecycle, measured inference, and audio artifact
creation. This is the only layer that decides workflow order. It has no HTTP or
Angular concerns and depends on inference through `TTSInferenceEngine`.

### Model boundary

The model registry maps stable API identifiers to backend model metadata. The
loader owns runtime lifecycle and the load-once cache. Neither responsibility
belongs to an API controller or Angular component.

### Inference boundary

`TTSInferenceEngine` is the replaceable text-to-speech port.
`KokoroONNXEngine` is its first concrete implementation. A different runtime or
model family must be adoptable by implementing the port and changing backend
composition, without changing Angular or the public synthesis contract.

### Audio and metrics boundaries

The audio service owns encoding, duration, storage, and externally addressable
audio artifacts. The metrics collector owns inference latency, exact generated
duration, process RSS memory, real-time factor, and cold/warm classification.
The collector times the model-loader boundary; the loader reports reuse state.
Inference adapters return raw audio; they do not calculate or format metrics.

### Measurement definitions

- Model load timing covers the cold loader boundary, including artifact
  validation and runtime initialization, and is zero for warm cache reuse.
- Inference timing starts immediately before `TTSInferenceEngine.synthesize`
  and ends when raw audio returns. Model loading, encoding, and storage are
  excluded.
- Audio duration is derived from `sample_count / sample_rate`.
- RTF is `inference_time / audio_duration`.
- Memory is process resident set size measured after inference.
- Warm means the engine existed before the current request; it does not mean a
  previous response was returned. Every synthesis request executes inference so
  every metric snapshot belongs to that request.

### Benchmark boundary

The benchmark runner executes a predefined corpus through the same application
services used by interactive synthesis. The evaluator aggregates individual
measurements. Benchmark code must not create a second, behaviorally different
inference path.

## Primary synthesis request flow

The required request path is:

```text
POST /api/synthesis
         │
         ▼
API Controller
         │
         ▼
SynthesisService
    ┌────┼──────────────┐
    ▼    ▼              ▼
 Model  Inference     Metrics
Registry Adapter     Collector
         │
         ▼
    Kokoro ONNX
         │
         ▼
    Audio Service
         │
         ▼
  SynthesisResult
```

In implementation terms, the registry resolves metadata, `ModelLoader` creates
one cached engine and reports its load lifecycle, `MetricsCollector` measures the
abstract inference call, and the audio service writes a stable request-addressed
WAV referenced by `SynthesisResult`.

## Dependency rules

1. Frontend feature components depend on frontend API abstractions and shared
   frontend contracts, never backend implementation modules.
2. Backend API controllers depend on Pydantic contracts and application
   services, never concrete inference adapters.
3. `SynthesisService` owns sequencing across the model, measured inference, and
   audio boundaries; measurement logic never enters controllers or adapters.
4. Concrete inference adapters depend inward on the inference abstraction's
   contracts; no domain or controller depends on Kokoro ONNX directly.
5. Benchmark execution reuses synthesis orchestration and metric semantics.
6. Model weights, generated audio, and benchmark output are runtime artifacts
   and are excluded from Git.

## Composition and future work

`backend/app/main.py` composes routers, services, the registry, the cached model
loader, the Kokoro engine factory, and local audio delivery. Angular consumes
the resulting contracts without backend imports. Benchmark execution,
production retention, and readiness policy remain deliberately deferred.

The implementation sequence and evidence gates are maintained in
[`ITERATIVE_CODING_MAP.md`](ITERATIVE_CODING_MAP.md).
