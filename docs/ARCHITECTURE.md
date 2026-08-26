# Architecture

## Purpose

OpenVoice Lab is split into an Angular presentation client and a FastAPI
application boundary. The repository begins with stable component boundaries so
that model runtimes and benchmarking mechanics can evolve without leaking into
the browser or HTTP controllers.

Stage 2 runs Kokoro v1.0 locally through ONNX Runtime. It loads the model once,
generates and serves WAV artifacts, and keeps the Stage 1 API shape intact. The
repository does not yet collect performance measurements or execute benchmarks.

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
```

This dependency direction is now executable through Kokoro and local WAV output.
ONNX and Kokoro remain behind the backend inference abstraction.

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
TTSInferenceEngine
  ↓
KokoroONNXEngine
  ↓
AudioService
  ↓
Typed JSON response + local WAV
```

### Synthesis application boundary

`SynthesisService` owns the complete synthesis workflow. It currently
coordinates model resolution/lifecycle, an inference engine, result reuse, and
audio artifact creation; metrics join this sequence next. This is the only layer
that decides workflow order. It has no HTTP or Angular concerns and depends on
inference through `TTSInferenceEngine`.

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
audio artifacts. The metrics collector owns inference latency, process memory,
real-time factor, and cold/warm classification. Inference adapters return raw
audio; they do not format API responses.

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
one cached engine, `SynthesisService` invokes the abstract inference port, and
the audio service writes a stable request-addressed WAV referenced by
`SynthesisResult`. Metrics will wrap this path in the next stage.

## Dependency rules

1. Frontend feature components depend on frontend API abstractions and shared
   frontend contracts, never backend implementation modules.
2. Backend API controllers depend on Pydantic contracts and application
   services, never concrete inference adapters.
3. `SynthesisService` owns sequencing across the model, inference, and audio
   boundaries; metrics will join that orchestration without entering controllers.
4. Concrete inference adapters depend inward on the inference abstraction's
   contracts; no domain or controller depends on Kokoro ONNX directly.
5. Benchmark execution reuses synthesis orchestration and metric semantics.
6. Model weights, generated audio, and benchmark output are runtime artifacts
   and are excluded from Git.

## Composition and future work

`backend/app/main.py` composes routers, services, the registry, the cached model
loader, the Kokoro engine factory, and local audio delivery. Metrics, benchmark
execution, production retention, and readiness policy remain deliberately
deferred.

The implementation sequence and evidence gates are maintained in
[`ITERATIVE_CODING_MAP.md`](ITERATIVE_CODING_MAP.md).
