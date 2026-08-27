# Architecture

## Purpose

OpenVoice Lab is split into an Angular presentation client and a FastAPI
application boundary. The repository begins with stable component boundaries so
that model runtimes and benchmarking mechanics can evolve without leaking into
the browser or HTTP controllers.

Stage 12 adds an independently composed SpeechT5 experiment surface beside the
Kokoro product. Angular still knows HTTP contracts only. Verified Stage 11
artifacts drive the training report, while a replaceable CPU runtime owns live
SpeechT5/Whisper execution. The experiment never enters `SynthesisService`, the
Kokoro registry, or the existing benchmark path.

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
TextProcessingService
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
├──────────────────────────────────────────────────────────────────┤
│ Text processing: optional normalization, then sanitization      │
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

The current synthesis page owns text and selection state, two independent
preprocessing toggles, model discovery, request lifecycle, and user-facing
recovery messages. It displays the final inference text when processing changed
the input but does not implement transformation rules. `model-selector` and
`audio-player` are presentational boundaries. `api` is the only frontend module
that performs HTTP calls. Local development proxies `/api` and `/audio` without
changing those public contracts.

### API boundary

FastAPI controllers are transport adapters. They may validate Pydantic request
contracts, invoke an application service, serialize its result, and map known
application errors to HTTP responses. They must not normalize or sanitize text,
choose or load models, run inference, encode audio, calculate RTF, inspect
process memory, or aggregate benchmark results.

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
TextProcessingService
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
coordinates text processing, model resolution/lifecycle, measured inference,
and audio artifact creation. This is the only layer that decides workflow
order. It has no HTTP or Angular concerns and depends on inference through
`TTSInferenceEngine`.

### Text-processing boundary

```text
Raw request text
  ↓
TextProcessingService
  ↓ optional TextNormalizer
  ↓ optional TextSanitizer
exact normalizedText
  ↓
measured inference
```

`TextProcessingService` owns order, optional execution, output limits, and
validation. `TextNormalizer` converts supported English currency, percentage,
URL, email, relative-path, Markdown, code, operator, snake-case, and camel-case
notation into speech-friendly text. `TextSanitizer` then owns deterministic
NFKC canonicalization, whitespace/control cleanup, and remaining punctuation-noise
rules. Either step can be bypassed independently; both disabled means the
validated original is unchanged. The original request remains available as
`text`, while `normalizedText` is the exact engine input and artifact-key input.
Processing occurs before model loading and outside `inferenceMs`. A sanitized
result with no speakable alphanumeric content is a domain error mapped to `422`.

### Model boundary

The model registry maps each stable API identifier to one complete backend
configuration: precision, artifact path, voices, runtime metadata, and inference
settings. `kokoro-fp32` and `kokoro-q8` use the same engine abstraction while
maintaining separate cached runtime sessions. The loader owns runtime lifecycle
and the load-once cache. Neither responsibility belongs to an API controller,
Angular component, or conditional branch in `SynthesisService`.

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

The executable benchmark flow is:

```text
sentences.json + SHA-256
  ↓
benchmark coordinator
  ├── fresh FP32 worker ──> SynthesisService ──> raw results
  └── fresh INT8 worker ──> SynthesisService ──> raw results
  ↓
corpus-hash verification
  ↓
BenchmarkEvaluator
  ↓
immutable timestamped JSON
```

Model processes are isolated because process RSS is not comparable if the
second model shares a process with the first model's cached runtime. Each worker
records the corpus version and hash, model ID, precision, voice, case identity,
original corpus text, final inference text, both preprocessing states, raw
metrics, audio URL, and any exception. The coordinator rejects a worker whose
corpus hash or text-processing configuration differs. The evaluator never
removes failed cases; it reports success and failure counts and aggregates only
successful measurements.

Aggregate semantics:

- Average and median latency use `inferenceMs` from successful cases.
- P95 latency uses deterministic linear interpolation over successful cases.
- Average RTF is the mean of per-case RTF values.
- Memory includes average and peak process RSS from the isolated model worker.
- Failure count includes every case without a synthesis result.

### Benchmark product boundary

The browser workflow is:

```text
Angular benchmark page
  ↓ GET /api/benchmarks/config
fixed corpus and model counts
  ↓ POST /api/benchmarks
BenchmarkJobService → background isolated coordinator
  ↓ GET /api/benchmarks/{id}
progress snapshots → completed BenchmarkResult
  ↓
Angular comparison table
```

`BenchmarkJobService` owns in-memory job identity, background task lifecycle,
progress snapshots, failure state, and latest-job recovery. HTTP controllers
only validate, call the service, and serialize responses. Angular displays
public benchmark contracts and does not read result files, invoke Python, or
know about worker processes beyond user-facing progress language.

The table deliberately limits presentation to average and p95 latency, average
RTF, peak RSS, and failure count. Detailed raw cases, environment data, corpus
hash, and audio URLs remain in `BenchmarkResult` for engineering use.

### SpeechT5 experiment boundary

Stage 11 training and Stage 12 presentation are separate responsibilities:

```text
verified Stage 11 artifacts ──> ExperimentReportService ──> read-only report API
locked shared test manifest ──> ExperimentFixtureService ──> fixture API

Angular Experiment tab
  ↓ POST comparison
ExperimentJobService (durable, bounded queue)
  ↓ normalize → sanitize
ExperimentModelRegistry
  ↓ replaceable ExperimentInferenceRuntime
SpeechT5 CPU + pinned speaker profile + pinned vocoder
  ↓ progressive WAV
pinned Whisper ASR → exact term scorer + WER
  ↓
atomic job/result/manifest files
```

`ExperimentReportService` fails closed if the final audit, dataset lock,
selected model, source revision, configuration hash, or artifact manifest is
inconsistent. Training statistics are never copied into Angular constants.
Historical RTX 4090 results and live CPU results use distinct contracts and
labels because they are not directly comparable.

The experiment model registry is immutable and separate from the deployable
Kokoro registry. It resolves public IDs for the pinned pretrained model and the
three selected Stage 11 checkpoints without leaking paths to Angular. Live
models execute sequentially against one final text and one speaker profile; an
LRU cache bounds process memory while reporting cold/warm state honestly.
The shared vocoder and ASR evaluator are loaded before per-model measurement.
Live memory is labeled as a process RSS snapshot because it still includes the
bounded model cache; the isolated Stage 11 evaluation remains the deployment
comparison for model-specific memory.

`ExperimentJobService` owns preprocessing order, progress, cancellation,
partial failures, recovery after process restart, and durable output. Each job
stores its request, latest status, generated WAVs, terminal result, provenance,
and a SHA-256 manifest below the ignored Stage 12 artifact root. Controllers
only validate, call the service, map errors, and stream snapshots over SSE.

The ASR transcript is an evaluation proxy, not a ground-truth pronunciation
claim. Domain-term accuracy checks complete normalized token sequences, WER is
deterministic Levenshtein distance, and both remain visible beside the playable
audio so a visitor can challenge the automated result.

### Deployment boundary

```text
docker compose up
  ↓
model-init (one shot)
  ↓ download + SHA-256 verification
model-artifacts named volume
  ↓ service_completed_successfully
FastAPI backend ── health check ──> healthy
  ↓
Angular/Nginx frontend ── /api + product/experiment audio proxy ──> FastAPI
```

Model weights never enter Git, either image build context, or an image layer.
The initializer downloads them from the documented upstream release into a
named volume and refuses checksum mismatches. Generated WAVs and benchmark JSON
use separate persistent volumes. The backend container runs as an unprivileged
user; Nginx is the browser-facing same-origin boundary.

Compose startup ordering is evidence based: the backend waits for successful
artifact verification, and the frontend waits for backend health. Base images
are pinned by immutable manifest digest, Angular installs from its lockfile,
and Python installs from a container-specific runtime lock. Deployment
configuration does not expose model paths or ONNX details to Angular.

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
   audio boundaries; measurement and model-variant branching never enter
   controllers or adapters.
4. Concrete inference adapters depend inward on the inference abstraction's
   contracts; no domain or controller depends on Kokoro ONNX directly.
5. Benchmark execution reuses synthesis orchestration and metric semantics;
   every model receives the same hashed corpus and failures remain raw data.
6. Model weights, generated audio, and benchmark output are runtime artifacts
   and are excluded from Git.
7. Containers preserve application boundaries: provisioning owns external
   weights, FastAPI owns inference, and Nginx owns static delivery and proxying.
8. Text transformation belongs to `text_processing`; Angular chooses policy,
   the synthesis service sequences normalization before sanitization, and
   inference adapters receive only the final text.
9. The SpeechT5 experiment registry/runtime remain separate from the Kokoro
   product registry/runtime; shared code is limited to technology-neutral text
   processing and HTTP/error conventions.
10. Training reports are projections of verified artifacts. Angular never owns
    training metrics, model paths, hashes, or evaluation math.

## Composition and future work

`backend/app/main.py` declaratively composes routers, services, both registry
definitions, the cached model loader, the Kokoro engine factory, local audio
delivery, and the benchmark job service. The CLI and browser job coordinator
both compose one fresh application per model worker and access the same
`SynthesisService`. It also conditionally composes the artifact-backed Stage 12
experiment service when optional CPU dependencies, pinned models, selected
Stage 11 weights, and the speaker profile are present. Missing experiment assets
do not break Kokoro startup; live experiment requests fail explicitly while
read-only evidence remains available. Compose reproduces the Kokoro product
runtime on Linux; Stage 12 CPU packaging, multi-replica experiment coordination,
and retention remain deliberately deferred.

The implementation sequence and evidence gates are maintained in
[`ITERATIVE_CODING_MAP.md`](ITERATIVE_CODING_MAP.md).
