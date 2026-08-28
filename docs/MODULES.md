# Module responsibilities

The tables below are ownership rules. A module may call another module through
its public contract, but it should not absorb the other module's responsibility.

## Frontend

| Module | Responsibility |
| --- | --- |
| `synthesis` | Text, preprocessing policies, voice/model selection, request workflow, and final-text preview. |
| `audio-player` | Playback and audio duration. |
| `metrics` | Display inference latency, RTF, memory and model metadata. |
| `benchmark` | Trigger benchmark runs and display aggregate results. |
| `experiment` | Present verified Stage 11 evidence and own fixture/custom live-comparison UI state. |
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
| `text_processing` | Ordered, optional English normalization, sanitization, and processed-text validation. |
| `models` | Model registry, model loading and lifecycle. |
| `inference` | Abstract TTS inference interface plus Kokoro ONNX and pinned SpeechT5 CPU adapters. |
| `audio` | Encoding, duration and audio-file handling. |
| `metrics` | Latency, process memory, RTF and cold/warm metrics. |
| `benchmark` | Predefined test execution and result aggregation. |
| `experiments` | Artifact verification/reporting, locked fixtures, experiment registry, scoring, durable comparison jobs, and orchestration. |
| `config` | Environment/model/deployment configuration. |
| `health` | Liveness/readiness support. |

## Deployment packaging

| Boundary | Responsibility |
| --- | --- |
| `backend/Dockerfile` | Digest-pinned, dependency-locked, non-root Python/ONNX runtime. |
| `frontend/Dockerfile` | Reproducible Angular production build and Nginx runtime. |
| Root `Dockerfile` | Single-container Cloud Run source build: Angular assets, FastAPI, verified Kokoro weights, and snapshot evidence. |
| `frontend/nginx.conf` | SPA delivery and same-origin proxy for API, Kokoro/experiment audio, and health. |
| `model-init` | Download and checksum-verify external model dependencies. |
| `docker-compose.yml` | Startup ordering, ports, health checks, and persistent volumes. |
| `.dockerignore` | Keep local dependencies, secrets, weights, and outputs out of image contexts. |

## Stage 12 implementation status

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
- Implemented deployment path: verified model volume → healthy FastAPI service
  → Angular/Nginx browser entry point.
- Implemented text-quality path: independent Angular policies →
  `TextProcessingService` → optional `TextNormalizer` → optional
  `TextSanitizer` → exact inference text.
- Implemented benchmark provenance: original text + final inference text +
  sanitizer/normalizer states in every raw case and the run configuration.
- Implemented Stage 11 training evidence: locked V1/V2/V3 schedules, 125-step
  recoverable checkpoints, final audit, selected-model hashes, validation
  histories, shared-test evaluation, cost/timing, and incident provenance.
- Implemented Stage 12 experiment path: lazy Angular tab → artifact-backed
  report/fixture/model APIs → bounded durable job service → replaceable CPU
  SpeechT5 runtime → progressive WAV → pinned Whisper → term/WER scoring.
- Implemented Cloud Run path: root source build → one non-root FastAPI process →
  API/static route separation → live Kokoro synthesis + verified read-only
  experiment snapshot.
- Current experiment presentation: one measured pretrained control → four V1
  update strategies on identical locked manifests → aggregate and live comparison.
- Implemented live-job durability: atomic request/status/result storage,
  cancellation, restart recovery, partial per-model failure, and terminal
  SHA-256 manifest.
- Still deferred: durable cloud experiment-job coordination and retention, live
  SpeechT5 cloud packaging, broader language-aware normalization, and
  statistically powered human evaluation.

## Boundary enforcement

- Technology names such as Kokoro and ONNX stay below the backend inference
  abstraction.
- HTTP serialization and status-code concerns stay in `api`.
- Cross-collaborator workflow order stays in `synthesis`.
- Normalization, sanitization, ordering, and processed-text validation stay in
  `text_processing`.
- Feature-specific frontend state stays in its owning feature; only genuinely
  reusable pieces move to `shared` or `core`.
- SpeechT5 experiment models stay in their own immutable experiment registry;
  the main synthesis registry exposes only the pinned pretrained control through
  the generic inference port, never adapted-run evidence or Whisper scoring.
- Historical training claims must be parsed from verified artifacts, and live
  CPU measurements must never be presented as historical RTX 4090 results.
