# OpenVoice Lab technical interview guide

This document is the deep technical learning guide for OpenVoice Lab. It explains
what the project does, why it was designed this way, how the major components
work, what the experiments actually measured, which decisions failed, and how
to defend the engineering trade-offs in an interview.

It is intentionally evidence-first:

- **Measured fact** means a value exists in a checked-in report or documented
  execution artifact.
- **Implemented behavior** means the repository contains the code and tests for
  the behavior.
- **Architectural intent** describes a boundary or future direction, not proof
  that a production-scale deployment already exists.
- **Hypothesis** is a plausible explanation that still requires another
  controlled experiment.

Use the guide with the source-of-truth documents linked throughout. The
[README](../README.md) is the concise portfolio narrative; this file is the
interview-depth explanation.

## 1. The project in one sentence

OpenVoice Lab is a self-hosted Angular and FastAPI system for synthesizing
speech with open-weight TTS models, measuring their serving performance, and
running reproducible SpeechT5 adaptation experiments against domain-specific
pronunciation metrics.

## 2. Interview-ready summaries

### 30-second version

> I built OpenVoice Lab to evaluate open-weight text-to-speech models as systems,
> not demos. Angular owns the user workflow, FastAPI owns validated contracts,
> a synthesis service orchestrates replaceable inference engines, and a model
> loader owns model lifecycle and memory. I added deterministic text processing,
> latency, real-time-factor and memory instrumentation, a fair benchmark runner,
> and a RunPod training pipeline with locked datasets and hash-verified
> checkpoints. The most important result was negative: the pretrained SpeechT5
> control beat most medical-domain adaptations, so the evaluation guardrails
> prevented me from promoting a worse model.

### Two-minute version

> The product has three Angular workflows: synthesis, model benchmarking, and an
> interactive SpeechT5 experiment. The frontend only knows API contracts. In the
> backend, controllers validate and translate HTTP; `SynthesisService` owns the
> application workflow; `ModelRegistry` describes available variants;
> `ModelLoader` owns expensive engine instances; and each runtime implements one
> `TTSInferenceEngine` interface. That keeps Kokoro ONNX, Audio8 ONNX, and
> SpeechT5 PyTorch details out of both Angular and the controllers.
>
> Before inference, a deterministic normalizer converts meaningful notation such
> as URLs, percentages, currency, and identifiers into speakable English. A
> sanitizer then removes remaining noise. The exact final text is returned as
> `normalizedText`, included in benchmark evidence, and used in deterministic
> audio identity. Inference timing excludes preprocessing and model load, so RTF
> remains interpretable.
>
> For ML work, I processed 6,661 raw WAV files into 6,628 approved 16 kHz mono
> samples, created leakage-safe locked splits, and trained SpeechT5 variants on
> separate secure RTX 4090 pods. Every run pinned the model revisions and dataset
> hashes, checkpointed optimizer/scheduler/RNG/trainer state, downloaded only
> complete hash-verified checkpoints, and terminated the pod after artifact
> verification. I compared the models using ASR-derived domain-term accuracy,
> WER, exact-sentence rate, latency, RTF, memory, and failure count. The
> pretrained control kept the best overall quality; a gradual-unfreeze model tied
> its term accuracy but regressed WER and runtime, while LoRA was the fastest
> adapted trade-off. That result demonstrates evaluation discipline rather than
> forcing a positive fine-tuning story.

### The strongest portfolio claim

OpenVoice Lab demonstrates ownership of the complete decision chain:

```text
raw data
  -> validated and locked manifests
  -> controlled training
  -> recoverable checkpoints
  -> identical evaluation workload
  -> product-visible evidence
  -> deployment decision
```

The claim is not that every trained model improved. The claim is that the system
can determine whether an adaptation deserves deployment.

## 3. Problem, users, goals, and non-goals

### Problem

Open-weight TTS models differ in pronunciation, naturalness, speed, memory use,
artifact size, runtime dependencies, and hardware behavior. Listening to a few
hand-picked clips cannot support a deployment decision. The project provides a
controlled path from text to audio plus comparable evidence.

### Primary use case

```text
text + model + voice + text-processing options
  -> generated WAV
  -> inference metrics and model metadata
```

### Secondary use cases

- Compare all locally available synthesis variants on the same corpus.
- Inspect cold versus warm model behavior.
- Evaluate medical-term pronunciation after SpeechT5 adaptation.
- Let a portfolio visitor run a fixture or custom-term comparison without using
  the command line.
- Reproduce training and artifact recovery on rented GPU infrastructure.

### Intended users

- An engineer selecting a model/runtime for deployment.
- An ML engineer testing an adaptation strategy.
- A reviewer or hiring manager examining system and experiment evidence.
- A learner studying full-stack AI system design.

### Explicit non-goals

- It is not a high-throughput, multi-tenant production TTS platform.
- It does not train Kokoro or Audio8.
- It does not use an LLM to normalize text.
- It does not call an external hosted TTS API.
- The current benchmark and experiment job stores are not durable distributed
  queues.
- The current ASR proxy is not a substitute for a statistically rigorous human
  listening study.

## 4. Architecture at a glance

### Product synthesis path

```text
Angular
  -> REST / JSON
FastAPI controller
  -> validated Pydantic schema
SynthesisService
  +-> TextProcessingService
  |     -> TextNormalizer -> TextSanitizer
  +-> ModelRegistry
  +-> ModelLoader
  |     -> TTSInferenceEngine
  |          +-> KokoroONNXEngine
  |          +-> Audio8ONNXEngine
  |          `-> SpeechT5 CPU engine
  +-> MetricsCollector
  `-> AudioService
        -> WAV + SynthesisResult
```

### Training and evaluation path

```text
immutable raw audio
  -> deterministic data processing
  -> audited, leakage-safe manifests
  -> locked V1/V2/V3 schedules
  -> pinned SpeechT5 training profile
  -> secure RTX 4090 pod
  -> checkpoint + SHA-256 manifest + complete marker
  -> immediate verified local download
  -> shared 662-case evaluation
  -> Whisper transcript
  -> domain-term accuracy + WER + serving metrics
  -> committed report snapshot
  -> Angular experiment interface
```

### Why the layers exist

| Layer | Owns | Must not own |
| --- | --- | --- |
| Angular | Interaction, form state, loading/error UI, playback, metric presentation | ONNX, model paths, Python classes, normalization rules |
| FastAPI controller | HTTP parsing, schema validation, status-code translation | Inference, model selection logic, orchestration |
| `SynthesisService` | End-to-end synthesis use case | Engine-specific APIs |
| Text processing | Deterministic English cleanup and normalization | Kokoro/SpeechT5 behavior |
| Registry | Immutable model/voice/runtime metadata | Loaded runtime instances |
| Loader | Engine creation, reuse, eviction, lifecycle | HTTP or UI decisions |
| Inference adapter | Runtime-specific synthesis | API response construction |
| Audio service | PCM/WAV encoding and stable artifact handling | Model execution |
| Metrics collector | Load/inference timing, duration, RTF, process memory | Product orchestration |
| Benchmark/evaluation | Equal workloads and aggregate evidence | Hidden cherry-picking |

The central rule is dependency direction: product logic depends on the inference
abstraction, while concrete runtimes depend on that abstraction. Angular never
depends on a concrete model implementation.

## 5. Repository map

```text
OpenVoice-Lab/
  frontend/                 Angular application
    src/app/synthesis/      Main generation workflow
    src/app/model-selector/ Dynamic model and voice selection
    src/app/audio-player/   Playback and duration
    src/app/metrics/        Inference metric presentation
    src/app/benchmark/      Benchmark jobs and comparison table
    src/app/experiment/     Stage 11 evidence and live comparison
    src/app/api/            Typed HTTP clients
    src/app/core/           Configuration and global concerns
    src/app/shared/         Reusable presentation/state
  backend/
    app/api/                Thin controllers
    app/schemas/            Pydantic request/response contracts
    app/synthesis/          Synthesis orchestration
    app/text_processing/    Normalizer, sanitizer, processing service
    app/models/             Registry and runtime lifecycle
    app/inference/          Replaceable engine implementations
    app/audio/              WAV encoding and files
    app/metrics/            Runtime measurement
    app/benchmark/          Corpus execution and aggregation
    app/experiments/        SpeechT5 report/live-comparison API
    app/config/             Typed environment configuration
    tests/                  Unit, API, orchestration and integrations
  data-processing/          Raw/processed data and generated manifests
  training/
    config/                 Frozen experiment profiles
    data_pipeline/          Data preparation
    dataset_variants/       V1/V2/V3 schedule construction
    full_training/          SpeechT5 train/evaluate/checkpoint logic
    runpod_agent/           Reusable remote-run toolkit
  docs/                     Architecture, contracts and evidence
  artifacts/                Ignored local run/checkpoint/model artifacts
  Dockerfile                Single-container Cloud Run build
  docker-compose.yml        Local multi-container environment
  start.ps1                 Local quick start
```

See [MODULES.md](MODULES.md) for the concise ownership table and
[ARCHITECTURE.md](ARCHITECTURE.md) for the normative boundaries.

## 6. Frontend design

### Technology and composition

The frontend uses Angular 20, TypeScript, standalone components, Angular forms,
the router, RxJS, and `HttpClient`. Components use an explicit state model for
idle, loading/running, success, and failure. The experiment route is lazy loaded
because it contains richer reporting and live-comparison UI than the primary
synthesis page.

The three top-level product areas are:

1. **Synthesis**: enter text, select an API-provided model and compatible voice,
   choose sanitization/normalization, synthesize, play audio, and inspect
   metrics.
2. **Benchmarks**: start a reproducible corpus run, monitor it, and compare
   aggregate performance across every currently synthesis-ready model.
3. **Experiment**: understand the SpeechT5 training evidence and interactively
   compare the pretrained control with four V1 adaptation approaches.

### Why model selection is API-driven

`GET /api/models` is the source of truth. Angular does not hard-code a branch
such as `if model === kokoro`. The response supplies identifiers, names,
precision/runtime metadata, compatible voices, availability, and unavailable
reason. This matters because:

- Model availability depends on provisioned artifacts and dependencies.
- Kokoro uses voices such as `af_heart`; Audio8 uses `unconditioned`; SpeechT5
  uses a pinned CMU speaker profile exposed as `cmu-slt`.
- Adding a registry entry should not require inference knowledge in a component.
- Disabled choices can explain infrastructure limits without pretending to run.

### Frontend boundary answer

If asked, “Why not put normalization or model rules in Angular?” answer:

> The backend is the contract and execution authority. A second client, a curl
> request, or a benchmark must produce the same final inference text and validate
> the same voice compatibility. Putting rules in Angular would duplicate
> behavior and make results client-dependent. Angular sends options and renders
> `normalizedText`; it does not implement the rules.

### Live experiment interaction

The Stage 12 interface supports fixture and custom modes. A custom case requires
explicit target terms, because domain-term accuracy has no denominator without
them. A comparison job emits server-sent events for progress and also supports
polling as a fallback. Models run sequentially on the same CPU and speaker
profile to limit memory and preserve comparability.

Typical states are:

```text
queued -> preprocessing -> loading -> synthesizing -> audio_ready
       -> transcribing -> scoring -> completed
```

Cancellation is cooperative: an active blocking model operation completes
before the job observes cancellation. This is safer than killing a shared
runtime in an unknown state.

## 7. API and backend composition

### Public contracts

The core API is:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness for local/Cloud Run health checks |
| `GET /api/models` | Current model catalog and availability |
| `POST /api/synthesis` | Generate audio plus metrics |
| `POST /api/benchmarks` | Start the complete benchmark |
| `GET /api/benchmarks/config` | Corpus and model configuration |
| `GET /api/benchmarks/{id}` | Job state/result |
| `GET /api/experiments/stage11/report` | Verified training report snapshot |
| `GET /api/experiments/stage11/fixtures` | Locked interactive cases |
| `GET /api/experiments/stage11/models` | Live experiment model availability |
| `POST /api/experiments/stage11/comparisons` | Start fixture/custom comparison |
| `GET .../{jobId}/events` | Stream progress with SSE |

Pydantic models use a shared alias generator: Python stays idiomatic with
`snake_case`, while JSON uses `camelCase`. `populate_by_name` allows internal
construction by Python field name without changing the public contract.

See [API.md](API.md) for payloads, status codes, and response fields.

### Thin-controller rule

A synthesis controller performs only:

```text
HTTP request
  -> Pydantic validation
  -> service call
  -> response or mapped HTTP error
```

It does not load a model, inspect ONNX, clean text, time inference, or encode
audio. This keeps controllers easy to test and prevents framework concerns from
becoming product logic.

### Error semantics

The service layer raises domain-specific errors and the API translates them.
The important categories are:

- `404`: unknown model, fixture, or job.
- `422`: unsupported voice or invalid/no-speakable processed text.
- `429`: experiment queue capacity reached.
- `503`: model artifacts/dependencies or experiment evidence unavailable.
- `500`: unexpected inference, audio, or metrics failure.

FastAPI/Pydantic also returns `422` for malformed request schemas before the
controller calls the service.

### Blocking inference in an async server

Local TTS execution is CPU/GPU-bound and synchronous. `SynthesisService` sends
the synchronous workflow through `asyncio.to_thread`, preventing that work from
directly blocking the event-loop thread. This is adequate for the deliberately
low-concurrency portfolio service, but it is not a replacement for a dedicated
worker queue at scale.

## 8. Complete synthesis request flow

For `POST /api/synthesis`, the service does the following:

1. Preserve the raw request text.
2. Run `TextNormalizer` when `normalizeText` is enabled.
3. Run `TextSanitizer` when `sanitizeText` is enabled.
4. Reject an empty/no-speakable result when sanitization is active.
5. Resolve immutable model metadata from `ModelRegistry`.
6. Build deterministic artifact identity from final text, model, voice,
   language, speed, and synthesis settings.
7. Ask `ModelLoader` for a cached or newly created engine.
8. Measure model-load state and time separately.
9. Measure only the concrete engine synthesis call as inference time.
10. Encode the returned samples to a mono WAV through `AudioService`.
11. Return original text, exact inference text as `normalizedText`, audio URL,
    model metadata, and metrics.

The exact processing order is important:

```text
normalization first -> sanitization second -> inference
```

If sanitization ran first, `$25` might become `25` before the normalizer could
produce `25 dollars`, and `15%` might lose the information needed to produce
`15 percent`.

When an inference error occurs after preprocessing, the service attaches the
final processed text to the error. That allows benchmark evidence to retain the
exact input that reached, or was intended for, inference rather than silently
falling back to raw text.

## 9. Text processing

### Sanitization versus normalization

These are separate operations with separate default-enabled API/UI toggles.

| Sanitizer | Normalizer | Result |
| --- | --- | --- |
| On | On | Expand supported meaning, then remove residual noise |
| On | Off | Remove noise/raw unsupported symbols without semantic expansion |
| Off | On | Expand supported notation and preserve other raw characters |
| Off | Off | Send the validated original text unchanged |

`normalizedText` always means “the exact final text sent to inference,” even if
neither operation changed the input.

### Normalizer rules

The normalizer is deterministic, English-focused, and uses explicit regexes plus
Python standard-library URL parsing. Its ordered rules cover:

- `$25` -> `25 dollars`; `$1` -> `1 dollar`.
- `15%` -> `15 percent`.
- `dev.team@example.com` -> `dev dot team at example dot com`.
- `https://example.com/docs` -> `example dot com slash docs`.
- `./models/kokoro.onnx` -> `models slash kokoro dot onnx`.
- Markdown links retain their label; emphasis markers are removed.
- Inline code markers are removed and identifiers are made speakable.
- `x == 5` -> `x equals 5`.
- `model_registry` -> `model registry`.
- `audioDuration` -> `audio duration`.

There is no LLM, external service, or model-specific exception in this layer.
Repeated normalization should be idempotent.

### Sanitizer rules

The sanitizer applies Unicode NFKC and whitespace cleanup, removes control and
zero-width characters, removes isolated `./`, collapses repeated noise such as
`---`, `,,,,`, `////`, and long dot runs, and removes stray `$`/`%` if the
normalizer did not consume them. It preserves ordinary English commas, periods,
apostrophes, questions, and grammatical hyphens.

It enforces a final length ceiling and rejects noise-only content instead of
sending an empty or meaningless request into the model.

### Known limits

- Rules are English-oriented.
- `$` means US dollars.
- Markdown handling is intentionally shallow, not a full parser.
- Arbitrary programming languages, mathematical notation, international phone
  formats, and other currencies require additional explicit rules.
- Determinism makes behavior testable, but a rule-based system cannot infer every
  context-dependent pronunciation.

## 10. Model catalog and replaceable inference

The product catalog currently describes five self-hosted configurations:

| Model ID | Runtime/precision | Voice contract | Main purpose |
| --- | --- | --- | --- |
| `kokoro-fp32` | Kokoro ONNX FP32 | `af_heart` and registry voices | Quality/reference variant |
| `kokoro-fp16` | Kokoro ONNX FP16 | Kokoro voices | Reduced precision comparison |
| `kokoro-q8` | Kokoro ONNX INT8 | Kokoro voices | Quantized CPU comparison |
| `audio8-0.6b` | Audio8 official INT4 ONNX CPU | `unconditioned` | Larger CPU-compatible open model |
| `speecht5-pretrained` | SpeechT5 FP32 PyTorch CPU | `cmu-slt` | Pretrained experiment control/product model |

All are local/self-hosted once artifacts are provisioned. None calls an external
inference API. Artifact provisioning is checksum-pinned. Availability is a
runtime fact: the registry can describe a model while reporting it unavailable
when its dependency or artifact marker is missing.

### Inference port

Every engine implements the same conceptual interface:

```python
class TTSInferenceEngine:
    @property
    def voices(self) -> tuple[str, ...]: ...

    def synthesize(
        self,
        text: str,
        voice: str,
        speed: float,
        language: str,
    ) -> AudioResult: ...
```

`AudioResult` contains mono float samples and a sample rate. The caller does not
need to know whether the engine used ONNX Runtime, PyTorch, a vocoder, or an
autoregressive decoder.

### Registry versus loader

This distinction is a common interview question:

- `ModelRegistry` owns immutable descriptions: ID, display name, precision,
  runtime, voices, artifact paths, and availability.
- `ModelLoader` owns mutable expensive state: live engine instances, thread-safe
  lookup, load count, LRU order, cleanup, and eviction.

Mixing them would turn a metadata endpoint into a lifecycle operation and make
simple catalog reads capable of loading hundreds of megabytes.

### Lifecycle and memory

`ModelLoader` uses a lock and an ordered LRU cache. A warm lookup returns the
existing engine; a cold lookup creates it once. When a configured cache limit is
exceeded, it closes and evicts the least recently used engine and requests
garbage collection. The Cloud Run profile keeps at most one product engine in
memory because the container has a 4 GiB target and several engines are roughly
1 GiB processes individually.

The design optimizes for predictable memory over zero switching latency. A
model switch may be cold, but concurrent model residency cannot silently exhaust
the instance.

### Runtime-specific notes

- **Kokoro** wraps a long-lived `kokoro_onnx.Kokoro` session and serializes
  access around the engine.
- **Audio8** uses its official INT4 ONNX CPU path and currently exposes an
  unconditioned voice; optional voice registration is not packaged.
- **SpeechT5** loads the pinned TTS model, HiFiGAN vocoder, and a pinned CMU
  speaker profile. It is CPU-compatible for the portfolio path; training used
  RTX 4090 GPUs.

## 11. Audio artifacts and deterministic identity

The audio service validates samples, clips them to `[-1, 1]`, converts them to
16-bit little-endian PCM, and writes a mono WAV. Audio duration is exact from:

```text
duration_seconds = sample_count / sample_rate
```

The artifact key hashes final inference text, model, voice, language, speed, and
other synthesis settings. The file name uses a shortened SHA-256 identity. This
means equivalent requests address the same intended artifact and preprocessing
changes cannot accidentally reuse raw-text audio.

The write path uses exclusive/atomic behavior so two equivalent requests do not
silently produce a partially overwritten WAV. Deterministic artifact identity
does not automatically prove bit-identical audio for every stochastic engine;
it defines request identity and safe file reuse.

## 12. Metrics and their formulas

### Model load time

Cold load measures engine creation. A warm cached engine reports zero model-load
time and `warm: true`. Load time is separate from inference because deployments
care about both cold-start behavior and steady-state request performance.

### Inference latency

Measured with a monotonic high-resolution timer around only
`engine.synthesize(...)`. It excludes preprocessing, controller overhead, model
load, WAV writing, network transfer, and UI work.

### Audio duration

Calculated from the generated sample count and sample rate, not wall-clock
guessing or browser metadata.

### Real-time factor

```text
RTF = inference time / generated audio duration
```

- `RTF < 1`: synthesis is faster than real time.
- `RTF = 1`: one second of compute per second of audio.
- `RTF > 1`: slower than real time.

The implementation rounds the reported inference and duration milliseconds and
then reports RTF to six decimal places. RTF is better than latency alone when
sentences produce different audio lengths.

### Memory

Product synthesis records process resident-set size through `psutil`. Training
evaluation also records peak GPU and process memory where available. RSS is a
process-level observation, not a perfect attribution of bytes to one model.

### Cold and warm state

“Warm” means the loader reused an already-created engine. It does not mean every
lower-level OS cache is warm, and it does not guarantee identical latency.

### Actual Kokoro warm example

One documented warm local request measured:

| Metric | Value |
| --- | ---: |
| Inference | 676.107 ms |
| Audio duration | 2,782.667 ms |
| RTF | 0.242971 |
| Process memory | 529.816 MB |

Check the [README Stage 4](../README.md#stage-4--instrument-inference-performance)
before quoting the environment or using the number as a cloud promise.

## 13. Reproducible model benchmarking

### Corpus design

The versioned benchmark corpus contains eight categories:

- short conversational
- medium conversational
- long-form
- numbers
- dates
- punctuation
- questions
- unusual names/words

Every selected model receives the same text cases in the same order. The
benchmark chooses each model's declared default voice rather than forcing an
incompatible Kokoro voice onto Audio8 or SpeechT5.

### Process isolation

Each model benchmark runs in its own subprocess. This gives a cleaner peak RSS
measurement and prevents the previous engine's resident memory/cache from
contaminating the next model. It also bounds a model crash to its worker.

### Aggregation

The evaluator records raw successes and failures, then calculates:

- average latency
- median latency
- p95 latency using interpolated position `(n - 1) * 0.95`
- average RTF
- average and peak process memory
- failure count

Failures are never removed from the report. Latency aggregates use successful
results, while failure count exposes reliability separately.

### Historic measured Kokoro comparison

An eight-case Windows 11/Python 3.13.5 CPU run produced:

| Variant | Avg latency | P95 | Avg RTF | Peak RSS | Failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kokoro FP32 | 1,273.603 ms | 2,571.653 ms | 0.217205 | 766.098 MB | 0 |
| Kokoro INT8 | 12,865.738 ms | 25,304.652 ms | 2.151270 | 601.859 MB | 0 |

The result is hardware/runtime-specific. Quantization reduced memory but was
roughly an order of magnitude slower in that environment. “Quantized” means
smaller numerical representation; it does not universally mean faster. Kernel
quality, dequantization overhead, operator support, CPU instruction sets, model
graph structure, and threading all matter.

### Benchmark limitations

- Eight cases are useful engineering smoke evidence, not a broad linguistic
  benchmark.
- Background jobs and results are in process/local storage.
- One Cloud Run instance is configured to keep that state coherent, but scale-to-
  zero or replacement loses it.
- Cross-hardware numbers must not be compared as if the environment were fixed.
- Naturalness is not captured by latency or RTF.

## 14. Medical speech data pipeline

### Immutable intake

The raw source was treated as immutable. The pipeline did not rewrite, move, or
delete original audio. It identified an identical duplicate raw root and
archived the duplicate rather than counting it as new data.

### Measured preparation outcome

| Item | Measured value |
| --- | ---: |
| Canonical raw WAV files | 6,661 |
| Raw bytes | 5,914,687,708 |
| Standardized readable audio | 6,661 |
| Approved samples | 6,628 |
| Rejected/excluded samples | 33 |
| Train unique rows | 5,303 |
| Validation rows | 663 |
| Test rows | 662 |

Audio was standardized to 16 kHz mono PCM. Hard checks covered duration,
clipping, RMS/silence, file readability, transcript validity, and identity.
Exclusion-reason counts can overlap because one sample can violate more than one
rule.

### Leakage prevention

The audit checks cross-split intersections across multiple identities, including
audio SHA-256, leakage groups, normalized/near transcript identities, and sample
IDs. All locked split intersection counts were zero. This is stronger than a
simple random row split because duplicated audio or near-identical transcripts
could otherwise leak into test data.

### Honesty about automated review

The first preparation run did not have ASR enabled. It recorded 6,628 ASR states
as `not_run`; it did not fabricate WER. Manual review was explicitly skipped:
1,843 flagged-but-not-hard-rejected items were accepted without pretending a
human approved each one. Hard validation remained active.

The initial local SpeechT5 processor/speaker-embedding readiness audit was also
blocked because the required Transformers dependency was absent. The split
manifests were still valid, but “training-ready” was not claimed until the remote
training environment completed the required preparation.

This is an important interview lesson: provenance should record missing checks,
not convert absence of evidence into a pass.

See [DATASET_PREPARATION.md](DATASET_PREPARATION.md) for the compact audit.

## 15. Training dataset schedules

The first Stage 11 experiment changed only the training schedule while keeping
validation/test data and model revisions fixed.

### V1 baseline

Uniform deterministic schedule over the approved training pool. Because batches
use blocks of eight, 5,303 unique train rows become 5,304 scheduled exposures;
one exposure repeats to fill the final block.

### V2 term balance

The builder increases exposure for supported medical terms while enforcing:

- at least 10 term utterances
- at least 3 speakers for an eligible term
- maximum row weight 4.0
- maximum 8 exposures per sample
- maximum speaker share 8%

The goal is to improve term representation without letting one speaker or one
recording dominate.

### V3 replay

Each eight-row block contains four term-balanced examples and four general
replay examples. The deterministic sampler can reorder complete blocks per
epoch, but it must not shuffle individual rows inside the locked block because
that would destroy the 50/50 contract.

Replay tests whether retaining general examples reduces catastrophic forgetting
while maintaining term exposure.

### Dataset lock

The lock records source manifest hashes, builder/config hashes, per-variant
manifests, row counts, and audit results. Training refuses a schedule that does
not match the lock. This prevents a comparison from silently changing its input
data after one model has already run.

## 16. SpeechT5 concepts required for the interview

### What SpeechT5 contributes

SpeechT5 is the trainable baseline used for the adaptation experiments. In this
project, a text-conditioned sequence-to-sequence model predicts acoustic
features, a speaker embedding conditions the voice, and a HiFiGAN vocoder turns
the acoustic representation into waveform audio.

The product inference abstraction hides those steps, but training needs to own
them explicitly.

### Pretrained versus adapted

- **Pretrained control**: the pinned Microsoft SpeechT5 weights with zero project
  training steps.
- **Adapted model**: starts from the same pinned weights and updates all or a
  subset of parameters using the medical speech manifests.

The control must be evaluated on the identical test manifest with the same
speaker/vocoder/ASR revisions. Otherwise an apparent gain could come from the
evaluator or serving path rather than training.

### Physical batch and gradient accumulation

Physical batch is the number of examples processed together in one forward and
backward pass. Gradient accumulation postpones the optimizer update while
adding gradients from multiple physical batches.

For one GPU:

```text
effective batch = physical batch * accumulation steps
```

Most runs used `16 * 2 = 32`. V1D used `8 * 4 = 32` because reduction factor 1
increases sequence work/memory. Matching effective batch makes update frequency
more comparable, although the different physical batch can still affect kernel
efficiency and normalization behavior.

### Epochs versus optimizer steps

An epoch is one scheduled pass over the training examples. An optimizer step
occurs after gradient accumulation. With 5,304 scheduled examples and effective
batch 32, one epoch is approximately:

```text
ceil(5304 / 32) = 166 optimizer steps
```

Therefore:

- 250 optimizer steps are about 1.51 epochs.
- 1,000 optimizer steps are about 6.02 epochs.

The report's exact epoch values come from the trainer/dataloader rather than
this approximation.

### BF16

BF16 reduces activation/gradient memory and improves throughput on supported
hardware while retaining FP32's exponent range. It has fewer mantissa bits, so
finite-value checks and stable scaling still matter. The runs used BF16, not
FP16.

### Gradient checkpointing

Gradient checkpointing stores fewer intermediate activations and recomputes
them during backward. It trades extra compute for lower memory, enabling larger
physical batches. It is unrelated to saving a recoverable model checkpoint;
the shared word “checkpoint” refers to two different mechanisms.

### Gradient clipping

The maximum gradient norm was 1.0. If the total norm exceeds that ceiling, the
gradients are rescaled before the optimizer update. This limits unstable spikes;
it does not repair bad data or guarantee convergence.

### Warmup and linear scheduling

The first 25 steps in the 250-step runs, or 100 steps in the 1,000-step runs,
warm the learning rate before a linear schedule. Warmup reduces the shock of
immediately applying the full learning rate to pretrained weights.

### Early stopping

At each evaluation interval, the trainer watches validation loss. Patience is
the number of evaluation rounds allowed without improvement beyond a threshold
before stopping. It protects compute and overfitting, but it can only optimize
the watched metric. Since validation loss was poorly aligned with domain
pronunciation quality here, early stopping alone could not select the best
deployable checkpoint.

### Reduction factor

SpeechT5 reduction factor is the number of acoustic frames predicted per
autoregressive decoder step. The source model uses factor 2. V1D changed it to
1, updated compatible output heads/configuration, and required more decoder
steps. This was a meaningful architectural experiment, not a simple scalar
speed knob. It performed very poorly and was rejected.

## 17. Reproducible training system

### Pinned inputs

Every approach pinned the same external revisions:

| Component | Revision |
| --- | --- |
| SpeechT5 TTS | `30fcde30f19b87502b8435427b5f5068e401d5f6` |
| HiFiGAN vocoder | `bb6f429406e86a9992357a972c0698b22043307d` |
| Speaker encoder | `56895a2df401be4150a159f3a1c653f00051d477` |
| Whisper ASR | `e8727524f962ee844a7319d92be39ac1bd25655a` |

Runs also record the Git commit, config SHA, dataset-lock SHA, run ID, pod ID,
seed, runtime image, and artifact hashes.

### RunPod agent toolkit

The reusable toolkit separates operations that should not be rewritten for each
experiment:

1. Create a secure, non-interruptible RTX 4090 pod and persist `pod.json`.
2. Build a run document that binds pod, profile, commit, dataset lock, and
   configuration.
3. Upload and start the pinned training entry point.
4. Poll provider state, process state, global step, loss, finite values, GPU
   memory, disk, checkpoints, evaluation, and cost.
5. Download checkpoints and final artifacts through a verified temporary path.
6. Terminate only after the final local verification gate passes.

Working documents live under:

```text
artifacts/stage11/agent-runs/<run-id>/
  pod.json
  run.json
  status.json
  events.jsonl
  checkpoint-inventory.json
  checkpoints/
  final/
```

The RunPod API key is read from `.env`; it is never printed, included in the run
bundle, command line, metadata, or Git.

### Checkpoint contract

A recoverable checkpoint contains model weights, optimizer state, scheduler
state, RNG state, trainer state, step, validation metrics, config/dataset hashes,
source revisions, run/pod identifiers, and a SHA-256 file manifest.

Completion is atomic:

1. Write all checkpoint files.
2. Write the manifest with every path, byte count, and SHA-256.
3. Write a completion marker containing the manifest hash.
4. Only then can the watcher consider the checkpoint downloadable.

Local download also uses a temporary directory. It verifies every file and then
atomically renames the directory into its final path. A corrupt or interrupted
download never replaces a valid local checkpoint, and the remote copy remains
available.

### Resume semantics

The trainer scans checkpoints newest-first and resumes only from the latest one
whose marker, manifest hash, file sizes, and file hashes all verify. A controller
failure is not automatically a training failure: the agent first checks the
remote PID and reattaches. It never silently launches a second trainer or
restarts at step zero.

### Pod termination gate

Termination is deliberately late:

```text
training finished
  -> selected/final model exported
  -> evaluation complete
  -> real synthesis reload check
  -> all required artifacts downloaded
  -> local SHA verification passed
  -> provider termination requested
  -> provider state confirmed TERMINATED
```

This converts expensive ephemeral GPU compute into durable, auditable evidence.

### Authorized stability-gate limitation

The earlier stability probe exercised physical batch 4, not physical batch 16.
The user explicitly authorized skipping a new batch-16 probe. Every applicable
run recorded that decision as `user_authorized_skip` or an approach-specific
override. The training completed, but the guide does not rewrite that waiver as
a prior successful batch-16 stability test.

See [TRAINING_AUTOMATION.md](TRAINING_AUTOMATION.md) for commands and recovery
controls.

## 18. How quality is evaluated

### Evaluation pipeline

```text
same expected sentence
  -> pretrained model audio
  -> adapted model audio
  -> pinned Whisper ASR transcript
  -> expected/transcript comparison
  -> domain terms + WER + exact sentence
  -> latency/RTF/memory/failures
```

### Domain-term accuracy

This is the headline domain metric:

```text
domain-term accuracy = correctly recognized target terms / total target terms
```

The scorer lowercases and tokenizes alphanumeric words plus internal
apostrophes. Each target term may contain multiple tokens; it is correct only
when the complete normalized token sequence appears contiguously in the ASR
transcript.

Example:

```text
targets: amlodipine, hypertension
ASR recognizes only hypertension
accuracy = 1 / 2 = 50%
```

This metric isolates the actual product concern that generic WER can hide.

### Word error rate

The repository implements word-level Levenshtein distance directly:

```text
WER = (substitutions + deletions + insertions) / reference word count
```

Lower is better. WER can exceed 100% if insertions are numerous. It measures the
whole sentence, so it complements rather than replaces term accuracy.

### Exact-sentence rate

The fraction of successful cases whose normalized ASR transcript exactly equals
the normalized reference. It is strict and can fall because of a single article
or punctuation-insensitive word difference.

### Serving metrics

Evaluation also records average inference latency, average RTF, peak GPU memory,
peak process memory, successful cases, and failures. A high-quality model that
cannot meet serving constraints might still be unsuitable.

### ASR-proxy limitations

The metric measures a chain: TTS output plus ASR recognition. Whisper can
misrecognize correctly pronounced audio, especially rare terms. Therefore:

- The fixed ASR revision makes relative comparisons more controlled.
- Audio and transcripts must remain available for investigation.
- A blinded human listening study is needed before a high-stakes production
  decision.
- Domain-term accuracy should be described as **ASR-derived proxy accuracy**, not
  direct phonetic ground truth.

An optional deeper follow-up is phoneme error rate using a fixed grapheme-to-
phoneme system, but it is not part of the current measured report.

## 19. First Stage 11 experiment: dataset strategy

The first full experiment trained V1 baseline, V2 term balance, and V3 replay for
up to 1,000 optimizer steps with physical batch 16, accumulation 2, effective
batch 32, BF16, learning rate `1e-5`, seed 42, and evaluation/checkpoint every
125 steps.

### Verified result

| Model | Best eval loss | Domain terms | WER | Avg inference | Avg RTF | Peak GPU | Failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pretrained control | n/a | **91.59%** | **11.10%** | 965 ms | 0.1838 | 950 MB | 0 |
| V1 baseline | 0.441642 | 33.41% | 70.84% | 1,080 ms | 0.1567 | 1,114 MB | 0 |
| V2 term balance | 0.445358 | 26.20% | 72.88% | 988 ms | 0.1566 | 1,050 MB | 0 |
| V3 replay | 0.444507 | 35.10% | 70.19% | 1,233 ms | 0.1819 | 1,125 MB | 0 |

All three reached 1,000 steps; no early stopping occurred. Twenty-four required
checkpoints verified. Total GPU time was 1.9886 hours and estimated cost was
about $1.47.

### Correct engineering decision

Reject all three adapted checkpoints. Their validation loss looked numerically
reasonable, but pronunciation quality collapsed on the shared test set.

### What the result proves

- More exposure to domain terms does not guarantee correct pronunciation.
- A lower teacher-forced acoustic validation loss is not equivalent to better
  autoregressive generation or ASR-derived term accuracy.
- Dataset scheduling alone could not overcome the adaptation failure.
- Pretrained controls and quality guardrails are mandatory.

### Plausible causes, not established facts

The following are hypotheses for later ablation, not claims proved by the run:

- `1e-5` and roughly six epochs may have updated the full model too aggressively,
  causing catastrophic forgetting.
- Multi-speaker training conditioning may not align with the fixed serving
  speaker profile.
- Transcript/audio quality or noisy medical labels may limit supervised signal.
- Acoustic loss may improve while stop-token behavior or intelligibility gets
  worse.
- Term annotations identify exposure, not phonetic correctness.

The follow-up experiment changed training strategy while holding the V1 dataset
constant to isolate these possibilities more cleanly.

### Operational incidents

One controller required a UTF-8 replacement-decoding repair, and the V3 pod had
an SSH interruption around step 730. The remote trainers continued; neither run
silently restarted or changed configuration.

The authoritative compact evidence is in [STAGE11_TRAINING.md](STAGE11_TRAINING.md).

## 20. Follow-up Stage 11 experiment: four V1 approaches

All four approaches used the same locked V1 schedule, shared validation/test
manifests, effective batch 32, 250-step ceiling, warmup 25, evaluation every 25
steps, gradient norm 1.0, gradient checkpointing, and seed 42.

### Approach definitions

#### V1A conservative full model

Updates the complete model at a much smaller `1e-6` learning rate. This tests
whether the original collapse was caused by overly aggressive full-model
adaptation.

#### V1B LoRA

Freezes base weights and trains rank-8 adapters on 96 attention projection
modules (`q_proj`, `k_proj`, `v_proj`, `out_proj`) with alpha 16, dropout 0.05,
and learning rate `5e-5`. It reduces the number of trainable parameters and aims
to preserve pretrained capabilities.

#### V1C gradual unfreeze

Starts with modal/TTS heads, then at step 50 adds the top two decoder blocks
(layers 4 and 5). The encoder and lower decoder remain frozen. Heads use `1e-6`;
new decoder groups use `5e-7`. This tests constrained capacity with a staged
optimization transition.

#### V1D reduction factor 1

Changes the model from two acoustic frames per decoder step to one, using
physical batch 8 and accumulation 4. It tests finer autoregressive resolution at
the cost of more decoder steps and compatible head changes.

### Verified shared-test comparison

| Model | Term accuracy | WER | Exact sentence | Avg inference | Avg RTF | Peak GPU | Failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pretrained control | **91.59%** (381/416) | **11.10%** | not reported | 965.45 ms | 0.18385 | 949.83 MB | 0 |
| V1A full | 89.18% (371/416) | 22.53% | 51.96% | 1,097.11 ms | 0.18925 | 1,058.05 MB | 0 |
| V1B LoRA | 90.14% (375/416) | 13.70% | **57.85% adapted** | **900.78 ms** | **0.16665** | 1,083.96 MB | 0 |
| V1C gradual | **91.59%** (381/416) | 13.78% | 57.25% | 1,036.17 ms | 0.19104 | 964.20 MB | 0 |
| V1D RF1 | 0.00% (0/348) | 98.83% | 0.00% | 4,675.61 ms | 0.32051 | 1,094.38 MB | 68 |

“Exact sentence” is unavailable for the pretrained snapshot and must not be
invented. V1D has only 348 evaluated target terms because 68 synthesis cases
failed; that smaller denominator is another reason to show failures beside the
percentage.

### Training and cost evidence

| Approach | Best step/loss | Train loss | Trainer time | Selected step | Selection status | Cost |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| V1A | 250 / 0.707688 | 0.854483 | 178.75 s | 25 | Eligible | $0.6746 |
| V1B | 250 / 0.706281 | 0.864957 | 108.07 s | 25 | No checkpoint met all guardrails | $0.6829 |
| V1C | 250 / 0.749014 | 0.922974 | 123.91 s | 25 | Eligible | $0.5538 |
| V1D | 225 / 5.227898 | 5.511744 | 284.53 s | 200 | No checkpoint met all guardrails | $1.2870 |

The four pods consumed 4.32194 total GPU hours and an estimated $3.1983. Forty
checkpoints and final artifacts passed integrity checks, and all pods were
confirmed terminated.

Trainer time is much shorter than billable pod time because provisioning,
dependency/model setup, quality probes for every checkpoint, full 662-case ASR
evaluation, artifact downloads, and teardown also consume pod time.

### Deployment conclusion

- Keep the **pretrained control** for best overall measured quality.
- **V1C** preserved domain-term accuracy but increased WER and runtime, so it
  does not justify replacing the control.
- **V1B** is the most interesting serving trade-off: fastest inference and RTF
  with a modest term/WER regression. It is a candidate only if latency is worth
  the quality cost.
- Reject **V1D**.

No adapted approach unambiguously beat the pretrained model. That is the honest
answer to “Did fine-tuning work?”

### Why selected step can differ from best validation step

Validation loss chooses the acoustically best teacher-forced checkpoint. The
bounded quality selector instead evaluates a fixed probe and prefers checkpoints
that pass failure, WER-regression, term-regression, short-output, and transcript-
length guardrails. It ranks eligible checkpoints by highest term accuracy, then
lowest WER, lowest short-output rate, and earliest step. If none pass, it marks
the best available model as ineligible rather than hiding the failure.

That is why V1A/V1C selected step 25 even though validation loss was lowest at
250, and why V1B/V1D explicitly report that no checkpoint met every guardrail.

### Compatibility-gate incidents

- LoRA had three pre-training compatibility probe failures.
- Reduction-factor 1 had one pre-training compatibility probe failure.

The reusable probes were repaired before training started. No model training was
restarted, and the final records retain the incidents rather than erasing them.

## 21. Why validation loss and deployment quality diverged

This is a high-value interview topic.

Training/validation loss measures how well the model predicts target acoustic
features under teacher forcing. Deployment generates autoregressively and is
judged after vocoding and ASR. Those are related but different objectives:

```text
validation loss:
reference history -> predicted acoustic frames vs target frames

deployment metric:
free-running decoder -> vocoder -> waveform -> ASR -> recognized terms/words
```

A small acoustic error can compound across decoder steps. Stop-token behavior
can create short/long outputs. A model can reduce average spectral loss while
damaging rare-term intelligibility. The pretrained model may also contain
linguistic knowledge that is overwritten even as it fits the local acoustic
distribution.

The design response is multi-objective selection, not pretending one loss is
the product KPI.

## 22. Stage 12 interactive experiment system

### Purpose

Stage 12 turns ML artifacts into a reviewer-facing product. It explains the
training pipeline, dataset, approaches, validation history, integrity, cost,
incidents, and measured result using compact expandable sections. It also lets a
visitor compare actual generated audio on a fixture or custom sentence.

### Snapshot versus live execution

The committed report snapshot is always available and SHA-verified. Large
adapted weights are deliberately not committed to Git or baked into the main
source history. For live cloud comparison, a model inventory specifies an HTTPS
origin, expected file sizes, and SHA-256 values.

The runtime downloads a selected model into a temporary cache directory,
verifies every file, writes a completion marker, and atomically promotes the
model. Interrupted or corrupt downloads never become available models.

If no remote origin is configured, the report remains useful and the UI marks
live adapted models unavailable. It does not claim snapshot evidence makes the
weights locally runnable.

### Fair live comparison

- Fixture mode uses a predefined sentence and target-term set.
- Custom mode requires user-supplied target terms.
- Two to five selected models run on the same host, with the same speaker and
  evaluator.
- Each produces a playable WAV, Whisper transcript, term score, WER, and runtime
  metrics.
- Queue capacity is bounded to prevent unbounded memory/compute demand.

### Current scaling limitation

Experiment jobs are in memory, the model cache is instance-local, and generated
audio is ephemeral. This works with Cloud Run maximum instances 1. A durable
multi-instance design needs object storage, a persistent job table/queue,
idempotent workers, and externally addressable progress events.

## 23. Docker and Cloud Run deployment

### Single-container cloud design

The root multi-stage Dockerfile:

1. Uses pinned Node 22 to install with `npm ci` and compile Angular.
2. Uses pinned Python 3.12 to download and SHA-verify model artifacts.
3. Builds a slim CPU runtime with pinned CPU PyTorch and locked dependencies.
4. Copies FastAPI, models, and Angular static output into one final image.
5. Runs as unprivileged UID/GID 10001.
6. Starts Uvicorn on Cloud Run's injected `PORT`.

FastAPI serves `/api`, `/health`, audio, and the Angular assets. A custom static
handler falls back to `index.html` for Angular client routes, but reserves API,
audio, health, docs, OpenAPI, and experiment-audio paths so an unknown API route
does not incorrectly return the SPA.

### Current Cloud Run profile

| Setting | Value | Reason |
| --- | --- | --- |
| Compute | 2 vCPU, 4 GiB | CPU-compatible product models with bounded cache |
| Timeout | 600 seconds | Long synthesis/benchmark requests |
| Billing | Instance-based | Background benchmark work can continue |
| Min instances | 0 | Portfolio cost control |
| Max instances | 1 | In-memory jobs and local audio stay coherent |
| Concurrency | 1 | Prevents simultaneous heavy model loads |

The current production image is CPU-oriented. Training used remote RTX 4090
pods. An L4 Cloud Run design would require a GPU runtime image and a separate
capacity/cost evaluation; it should not be implied by the present CPU image.

### Ephemeral storage

The container writes generated audio, benchmark output, model cache, and job
state under `/tmp/openvoice`. Cloud Run can replace or scale down the instance,
so those files are not durable.

Production evolution:

```text
audio/models/results -> Cloud Storage
job state             -> database or managed queue
workers               -> idempotent job consumers
progress              -> durable event/status channel
```

Only after that migration should maximum instances exceed one.

### Local Docker Compose

Compose uses a model-init job, backend, and Nginx-hosted frontend with named
volumes for model artifacts, audio, and benchmark results. Health-gated startup
prevents the frontend workflow from assuming the backend is ready.

See [CLOUD_RUN.md](CLOUD_RUN.md) for build-trigger and deployment settings.

## 24. Testing strategy

### Backend

The backend test suite separates fast deterministic tests from real-model
integrations. It covers:

- health, models, valid and invalid synthesis API contracts
- controller/service separation through fakes
- sanitizer and normalizer rules, idempotence, length/rejection behavior, and
  all four toggle combinations
- exact normalized text reaching fake inference
- load/warm state, loader reuse, thread safety, LRU eviction, and failure cleanup
- metrics and RTF mathematics
- benchmark scheduling, per-model voices, progress, failures, and aggregation
- experiment report/fixture/model endpoints, queueing, SSE/polling, scoring, and
  cancellation
- Kokoro, Audio8, and CPU model integrations when provisioned

Integration tests are marked because they require large local model artifacts.

### Frontend

Vitest/Angular component tests cover:

- synthesis form validation and independent text-processing toggles
- dynamic model/voice selection
- loading, backend-unavailable, inference-failure, and success states
- normalized-text preview and audio playback handoff
- metric rendering
- benchmark start/progress/failure/result comparison
- experiment report, loss chart, live model selection, fixture/custom flows,
  event/poll behavior, and result rendering

### Training and toolkit

Training tests validate config/lock integrity, deterministic block sampling,
finite-value checks, approach-specific trainable parameters, reduction-factor
save/reload compatibility, checkpoint manifests/markers, latest-valid resume,
run-document persistence, download verification, and termination gates.

### What tests cannot prove

- Unit tests cannot prove audio sounds natural.
- One integration WAV cannot prove broad pronunciation quality.
- Mocked RunPod tests cannot prove provider availability.
- A passing container test cannot prove Cloud Run latency or cost.
- ASR metrics cannot replace blinded human judgment.

## 25. Security and supply-chain controls

- `.env`, credentials, raw/model artifacts, checkpoints, and generated audio are
  ignored where appropriate.
- RunPod secrets never enter the source bundle or logs.
- External model revisions are commit-pinned rather than floating names.
- Provisioned artifacts and adapted model downloads use expected byte counts and
  SHA-256 verification.
- Temporary downloads are atomically promoted only after verification.
- The Cloud Run container uses an unprivileged user.
- Public model availability exposes reasons, not credentials or internal secret
  values.
- The experiment access token is a Pydantic `SecretStr` and should be injected
  from Secret Manager.
- Remote model code is not executed.

SHA-256 proves the bytes match the expected inventory. It does not prove the
artifact is safe or correctly licensed; provenance, dependency scanning, and
license review remain separate responsibilities.

## 26. Key design decisions and trade-offs

### Why FastAPI plus Angular?

FastAPI provides typed Python contracts close to model code; Angular provides a
structured typed UI with clear component/service boundaries. A single framework
could reduce repository complexity, but the split demonstrates a realistic
backend-owned inference contract and independently testable client.

### Why an abstraction with only a few methods?

The interface contains only capabilities needed by product orchestration.
Exposing ONNX sessions or SpeechT5 processors would leak concrete runtimes and
make replacements expensive. Too broad an abstraction becomes a least-common-
denominator framework; the current small port keeps it purposeful.

### Why an in-process loader rather than one model server per model?

For a portfolio-scale single instance, an LRU loader is simpler and cheaper. At
high traffic, dedicated model workers/services would isolate failures, scale
independently, and avoid cold switching.

### Why subprocesses for benchmark models but threads for one synthesis?

One interactive request benefits from reusing the process's warm engine. A
benchmark needs cleaner memory isolation across models and can afford worker
startup. Different workloads justify different isolation.

### Why deterministic rules instead of an LLM normalizer?

Rules are reproducible, auditable, fast, offline, and easy to test. An LLM could
cover more cases but add latency, cost, nondeterminism, privacy concerns, prompt
injection risk, and another model dependency before TTS.

### Why domain-term accuracy over generic WER?

The business failure is mispronouncing important medical terms. WER can remain
low by recognizing many easy words while missing `hydrochlorothiazide`.
Term accuracy makes the decision-specific error visible; WER preserves the
whole-sentence view.

### Why keep failed experiments in the UI?

Deleting V1D or hiding regressions creates survivor bias. Showing failure count,
loss curves, incidents, and selection status demonstrates that the system makes
deployment decisions from evidence.

### Why one Cloud Run instance?

Because job state, generated audio, and caches are local. One instance prevents
requests from landing on a different in-memory world. It limits throughput and
availability, so it is a coherent temporary constraint rather than a final
production architecture.

## 27. Known limitations and concrete next steps

| Limitation | Risk | Production-oriented next step |
| --- | --- | --- |
| In-memory job state | Lost on restart/scale-down | Persistent job DB and managed queue |
| Local ephemeral WAVs/results | Broken URLs and lost evidence | Object storage with signed/public URLs |
| Max instance 1 | Low throughput/single-instance availability | Stateless API plus scalable workers |
| CPU-heavy interactive models | Long latency and concurrency pressure | Separate GPU/optimized model service or async jobs |
| Eight-case general benchmark | Weak coverage | Larger versioned multilingual/domain corpus |
| ASR proxy evaluation | Evaluator bias | Blind listening panel and optional phoneme metrics |
| Fixed English text rules | Limited notation/languages | Locale-aware rule packs and explicit versioning |
| One speaker profile in live SpeechT5 comparison | Limited voice generalization evidence | Multi-speaker evaluation matrix |
| Medical data quality uncertainty | Fine-tuning regressions | ASR audit, transcript correction, speaker/term ablations |
| Model weights remote/on demand | Cold download and origin dependency | Versioned model registry/object storage/CDN cache |
| No request authentication/rate limit | Public cost/abuse risk | Auth, quotas, payload limits, rate limiting |

The highest-value next ML experiment is not “train longer.” It is a controlled
ablation: verify transcript/audio quality for term cases, align speaker
conditioning, use the conservative/LoRA strategies, evaluate frequent fixed
checkpoints early, and stop when domain quality regresses.

## 28. High-probability interview questions and strong answers

### Architecture

**Q: What is the most important boundary?**

The inference port. `SynthesisService` depends on a stable contract returning
audio samples, while ONNX/PyTorch/model-specific objects remain inside adapters.
That boundary also keeps Angular and controllers model-agnostic.

**Q: Why does orchestration belong in `SynthesisService`?**

It is the application use case that coordinates preprocessing, model resolution,
lifecycle, metrics, inference, audio, and response identity. Putting those steps
in controllers ties business flow to HTTP; putting them in an engine makes a
runtime responsible for product policy.

**Q: How would you add a new model?**

Implement `TTSInferenceEngine`, register immutable metadata and an engine factory
mapping in the loader's composition point, add artifact/dependency availability
checks, test synthesis/voice/error behavior, and let `/api/models` drive the UI
and benchmark dynamically. No controller or Angular model branch is required.

**Q: Is the Open/Closed Principle perfectly achieved?**

Not perfectly. A new concrete engine still needs a composition registration in
the loader. The important point is that change is centralized at the factory/
registry boundary rather than scattered through controllers, services, and UI.

### Backend and concurrency

**Q: Why use `asyncio.to_thread`?**

The engine calls are blocking. Offloading avoids occupying the event-loop thread.
It does not create unlimited safe parallelism; engine locks, container
concurrency 1, and bounded jobs still control heavy work.

**Q: What happens if two requests load the same model?**

The loader lock serializes cache lookup and creation so the expensive engine is
created once. Subsequent callers reuse the instance. Tests cover reuse and
thread-safety behavior.

**Q: Why return `503` for an unavailable registered model?**

The ID is valid, but required runtime capacity/artifacts are not ready. `404`
would incorrectly say the model does not exist; `503` says the service cannot
currently fulfill it.

### Metrics and benchmarking

**Q: Why is RTF useful?**

Latency scales with output length. RTF normalizes inference compute by generated
audio duration and directly answers whether synthesis is faster than playback.

**Q: Why was INT8 slower than FP32?**

Quantization reduced memory, but the observed CPU/runtime did not execute that
graph efficiently. Operator kernels, dequantization, graph structure, threading,
and CPU instructions determine speed. The project reports the measured platform-
specific result rather than assuming smaller precision means faster.

**Q: Is the benchmark fair?**

It fixes text/order, records failures, uses compatible declared voices, isolates
each model in a subprocess, and reports the environment. It is fair for the
defined engineering comparison, but the eight-case corpus is too small for a
broad quality claim.

### ML and data

**Q: Why did pretrained SpeechT5 outperform fine-tuned versions?**

The measured fact is the regression. Likely causes include catastrophic
forgetting, speaker-conditioning mismatch, noisy data, and loss/quality
misalignment, but those remain hypotheses. The follow-up lower-LR/LoRA/gradual
experiment isolated training strategy and recovered much of the quality; none
unambiguously beat the control.

**Q: Why isn't validation loss enough?**

It is teacher-forced acoustic prediction loss. Deployment is free-running
generation followed by vocoding and human/ASR interpretation. The objectives are
correlated but not identical, especially for rare terms and stopping behavior.

**Q: Why use LoRA?**

It limits trainable capacity, can reduce catastrophic forgetting, and is faster
to optimize. In this run it produced the fastest adapted inference result with a
small quality regression, but no checkpoint passed every configured guardrail.

**Q: What did gradual unfreezing show?**

Constrained adaptation preserved the pretrained term accuracy: heads learned
first, then only the top decoder blocks joined. Whole-sentence WER and runtime
still regressed, so matching one headline metric did not justify deployment.

**Q: Why did reduction factor 1 fail?**

It doubled the approximate autoregressive decoder work and changed compatible
output behavior. The model did not learn that configuration adequately in the
250-step experiment: 68 failures, zero recognized target terms among evaluated
cases, 98.83% WER, and much slower inference. The result rejects this setup; it
does not prove reduction factor 1 can never work.

**Q: How did you prevent data leakage?**

Locked split audits checked sample IDs, audio SHA identities, leakage groups,
and normalized/near transcripts across train, validation, and test. All cross-
split intersections were zero.

**Q: Why is domain-term accuracy an ASR proxy?**

Correctness is inferred from a fixed Whisper transcript, not direct phonetic
annotation. This provides scalable relative measurement, but human listening is
needed to separate TTS mistakes from ASR mistakes.

### Infrastructure and reliability

**Q: How do you know a downloaded checkpoint is complete?**

The remote trainer writes the completion marker last. The watcher verifies the
marker points to the manifest hash, then checks every file size and SHA before
atomically publishing the local directory.

**Q: What happens if the controller loses SSH?**

It treats orchestration and training as separate processes. It checks the remote
PID and run document, reattaches, and resumes monitoring. It starts from a
checkpoint only if training actually stopped, and only from the latest fully
verified checkpoint.

**Q: Why not commit model weights?**

They are large deployment artifacts with independent licensing/provenance and
would bloat Git history. The repository commits exact identities, revisions,
sizes, and hashes; object storage holds the bytes.

**Q: Is the Cloud Run service production-ready?**

It is a reproducible portfolio deployment, not a high-scale production system.
Single-instance state and ephemeral storage are explicit limitations. A
production version needs durable artifacts/job state, authentication, quotas,
and independently scalable workers.

### Behavioral and ownership

**Q: Tell me about a failure.**

Use the first Stage 11 run. Three training schedules completed successfully, but
all severely regressed pronunciation. Explain that you did not promote them,
added quality-probe guardrails, tried lower-risk adaptation strategies, retained
all evidence/incidents, and kept pretrained as the deployment choice.

**Q: What would you do differently?**

Run small quality probes before 1,000 full-model steps, audit term audio with ASR
and listening before training, align training and serving speaker conditioning,
start with LoRA/gradual unfreezing, and make human evaluation a formal artifact.

**Q: What are you most proud of?**

The system makes it difficult to tell an unsupported success story. Data locks,
pinned revisions, control evaluation, explicit missing metrics, failed-model
visibility, and hash-verified recovery turn experiments into auditable decisions.

## 29. Whiteboard drills

Practice drawing and explaining these without reading:

### Drill A: synthesis in five boxes

```text
Angular -> FastAPI -> SynthesisService -> Inference port -> WAV/metrics
```

Then add preprocessing, registry/loader, and concrete engines.

### Drill B: why a registry and loader are different

```text
registry = what can run
loader   = what is currently alive
```

Explain cold/warm, lock, LRU, and cache size one.

### Drill C: failed fine-tuning decision

```text
lower validation loss
  != better domain pronunciation
  -> fixed ASR probe + term/WER guardrails
  -> reject adapted model
```

### Drill D: production evolution

```text
current: API + jobs + files in one instance
future:  stateless API -> durable queue -> model workers -> object storage
```

### Drill E: failure-safe checkpoint transfer

```text
write -> manifest -> complete marker -> temp download -> verify -> atomic rename
```

## 30. Formula and terminology sheet

| Term | Meaning |
| --- | --- |
| TTS | Text-to-speech synthesis |
| Open-weight | Model weights can be obtained/hosted under their license; not necessarily open-source training data/code |
| Inference | Running a trained model to generate output |
| ONNX | Portable model graph format executed by a runtime |
| FP32/FP16/BF16/INT8/INT4 | Numeric representations with different precision, memory, and kernel behavior |
| Cold load | Engine must be created and weights/session loaded |
| Warm request | Existing engine instance is reused |
| RTF | Inference seconds divided by generated audio seconds |
| Physical batch | Examples in one forward/backward pass |
| Accumulation | Number of physical batches whose gradients contribute to one optimizer update |
| Effective batch | Physical batch x accumulation x GPU count |
| Epoch | One scheduled pass over training examples |
| Optimizer step | One parameter update after accumulated gradients |
| Teacher forcing | Training/validation uses ground-truth prior outputs as decoder context |
| Catastrophic forgetting | Adaptation damages useful pretrained behavior |
| LoRA | Low-rank trainable adapters while base weights remain frozen |
| WER | Word-level edit distance divided by reference word count |
| Domain-term accuracy | Correct ASR-recognized target terms divided by all target terms |
| p95 | Value below which approximately 95% of observations fall |
| RSS | Resident process memory |
| SHA-256 | Cryptographic digest used here for byte-integrity verification |
| Idempotent | Repeating an operation has the same intended effect |
| SSE | Server-sent events, one-way server-to-browser event stream |

## 31. Evidence discipline: what not to say

Avoid these claims:

- “Fine-tuning improved SpeechT5.” It did not improve the overall measured
  control result.
- “V1C beat pretrained.” It tied term accuracy and lost on WER/runtime.
- “LoRA passed the quality gate.” It was selected as best available, but no
  checkpoint met every guardrail.
- “The pretrained exact-sentence rate was X.” It was not reported.
- “The batch-16 stability probe passed.” It was explicitly waived.
- “ASR proves pronunciation.” It is a controlled proxy.
- “Quantization is faster.” It was dramatically slower in the measured Kokoro
  CPU environment.
- “Cloud Run stores generated files.” Its writable filesystem is ephemeral.
- “The service scales horizontally.” Its current state model requires one
  instance.
- “The current image uses an L4.” The current documented image is CPU-only.

Use precise language such as “measured on,” “implemented and tested,” “available
when provisioned,” “ASR-derived,” “estimated provider cost,” and “hypothesis.”

## 32. Recommended study path

1. Read this guide once end to end.
2. Rehearse the 30-second and two-minute versions aloud.
3. Draw the synthesis and training diagrams from memory.
4. Study the exact API in [API.md](API.md).
5. Review module boundaries in [ARCHITECTURE.md](ARCHITECTURE.md) and
   [MODULES.md](MODULES.md).
6. Memorize the direction, not every decimal, of both Stage 11 result tables.
7. Know three exact facts: 6,628 approved samples, 662 shared test cases, and
   91.59% pretrained domain-term accuracy.
8. Practice explaining why negative experiment results are valuable.
9. Review [TRAINING_AUTOMATION.md](TRAINING_AUTOMATION.md) until you can explain
   recovery and verification without tool names.
10. Review [CLOUD_RUN.md](CLOUD_RUN.md) and state its limitations before an
    interviewer has to discover them.

## 33. Run, inspect, and verify locally

### Fastest reproducible path

From a fresh clone with Docker installed:

```powershell
git clone https://github.com/Louis-Tr/OpenVoice-Lab.git
Set-Location OpenVoice-Lab
docker compose up
```

Open `http://localhost:4200`; the API is available on
`http://localhost:8000`. The first run downloads checksum-verified model
artifacts into a named volume. Stop without deleting those volumes using:

```powershell
docker compose down
```

### Native development path

After installing the documented Python/Node dependencies and model artifacts,
the root launcher starts both development servers:

```powershell
.\start.ps1
```

Validate prerequisites without starting processes:

```powershell
.\start.ps1 -CheckOnly
```

Angular uses its development proxy so `/api` and `/audio` keep the same browser
contract. In the single production container, FastAPI directly serves both the
compiled Angular application and the API.

### Useful inspection requests

```powershell
curl.exe http://localhost:8000/health
curl.exe http://localhost:8000/api/models
curl.exe http://localhost:8000/api/benchmarks/config
curl.exe http://localhost:8000/api/experiments/stage11/report
```

Example synthesis:

```powershell
$body = @{
  text = 'Save 15% at https://example.com. Price: $25.'
  modelId = 'kokoro-fp32'
  voiceId = 'af_heart'
  sanitizeText = $true
  normalizeText = $true
} | ConvertTo-Json

$request = @{
  Method = 'Post'
  Uri = 'http://localhost:8000/api/synthesis'
  ContentType = 'application/json'
  Body = $body
}
Invoke-RestMethod @request
```

Inspect `/docs` for Swagger UI and `/openapi.json` for the generated OpenAPI
schema. Do not benchmark the first cold request and present it as warm steady-
state performance; record environment, model, text/corpus, cold/warm state, and
date with every number.

### Verification commands

Backend tests run from `backend/` in the configured development environment;
frontend tests run through the package script:

```powershell
Set-Location backend
pytest -q

Set-Location ..\frontend
npm test
npm run build
```

Real-model integration tests require the provisioned artifacts and should not be
confused with portable unit tests. A clean Angular build proves compilation and
bundling, not that a remote model origin or Cloud Run service is healthy.

## 34. Source-of-truth index

| Topic | Primary source |
| --- | --- |
| Portfolio build story and measured examples | [README.md](../README.md) |
| Boundaries and request flows | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Module ownership | [MODULES.md](MODULES.md) |
| API contracts | [API.md](API.md) |
| Stage progression | [ITERATIVE_CODING_MAP.md](ITERATIVE_CODING_MAP.md) |
| Data preparation audit | [DATASET_PREPARATION.md](DATASET_PREPARATION.md) |
| First full Stage 11 evidence | [STAGE11_TRAINING.md](STAGE11_TRAINING.md) |
| Reusable RunPod workflow | [TRAINING_AUTOMATION.md](TRAINING_AUTOMATION.md) |
| Interactive experiment plan/verification | [STAGE12_IMPLEMENTATION_PLAN.md](STAGE12_IMPLEMENTATION_PLAN.md) |
| Cloud deployment | [CLOUD_RUN.md](CLOUD_RUN.md) |
| Current four-approach measured snapshot | `backend/app/experiments/snapshots/report.json` |
| Frozen first full-training config | `training/config/full_training.yaml` |
| Four follow-up profiles | `training/config/v1a_conservative_full.yaml` through `v1d_reduction_factor_1.yaml` |

When documentation and a generated measurement disagree, inspect the artifact,
its manifest, source commit, and code path. Never resolve disagreement by
choosing the more impressive number.
