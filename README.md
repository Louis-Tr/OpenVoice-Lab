# OpenVoice Lab

**Architecture first. Benchmarks before claims.**

OpenVoice Lab is a modular platform for evaluating and serving open-weight TTS
models.

> “I designed a modular architecture for evaluating and deploying open-weight
> TTS models before implementing the inference system.”

## Stage 0 — Architecture-first repository

| | |
| --- | --- |
| **Problem** | Evaluate and serve open-weight TTS models. |
| **Primary flow** | Text → model → audio + performance metrics. |
| **Target stack** | Angular, FastAPI, Python, ONNX, Kokoro, Docker. |
| **Current reality** | Boundaries, contracts, docs, and minimal bootstrapping only. No synthesis or benchmark engine yet. |

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

Stage 0 establishes frontend/backend separation, a replaceable inference port,
stable API contracts, explicit model-lifecycle ownership, benchmark boundaries,
and documentation conventions. Angular never depends on ONNX or Kokoro. API
controllers stay thin. `SynthesisService` owns orchestration.

## Repository

```text
frontend/   Angular shell and feature boundaries
backend/    FastAPI shell and backend domain boundaries
docs/       Architecture, ownership, API, and delivery roadmap
```

Reserved contracts:

- `POST /api/synthesis`
- `GET /api/models`
- `POST /api/benchmarks`
- `GET /health`

## Stage 0 acceptance

- [x] Git repository and `main` branch initialized.
- [x] Frontend and backend boundaries exist.
- [x] Every planned module has documented ownership.
- [x] Initial API surface is defined.
- [x] Architecture and iterative delivery path are documented.
- [ ] Real inference, model loading, metrics, and benchmarking.

## Read next

- [Architecture](docs/ARCHITECTURE.md)
- [Module responsibilities](docs/MODULES.md)
- [API surface](docs/API.md)
- [Iterative coding map](docs/ITERATIVE_CODING_MAP.md)

**Portfolio status:** valid now as a software architecture and design
repository. Runtime claims begin only after measured implementation evidence.

