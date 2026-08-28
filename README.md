# OpenVoice Lab

**Own the stack. Measure the result.**

OpenVoice Lab evaluates and serves open-weight TTS models through a modular
Angular + FastAPI architecture.

### Current synthesis catalog

The main Synthesis tab is driven entirely by `GET /api/models` and exposes five
concrete configurations: **Kokoro FP32**, **Kokoro FP16**, **Kokoro INT8**,
**Audio8 0.6B**, and **SpeechT5**. Kokoro's three verified ONNX files and the
pinned SpeechT5 CPU profile run locally. Audio8 remains visibly setup-gated
until its separate reviewed runtime and model package are provisioned; the UI
does not present it as ready prematurely.

This README is a cumulative build record. New stages extend the story; completed
stages stay visible as evidence of how the system evolved.

## Stage 0 — Architecture-first repository

> “I designed a modular architecture for evaluating and deploying open-weight
> TTS models before implementing the inference system.”

**Responsibility:** establish boundaries before writing inference code.

```text
Angular → REST → FastAPI → SynthesisService → Inference abstraction → Kokoro
```

Stage 0 established the Angular/FastAPI split, replaceable inference port, API
contracts, model-lifecycle ownership, benchmark boundaries, and documentation
conventions. The repository initialized cleanly with every planned module
documented and no runtime claims.

**Portfolio proof:** software architecture and API design.

## Stage 1 — Executable backend contract

> “I converted the architecture into a working FastAPI service while keeping
> inference implementation replaceable.”

**Responsibility:** prove the API boundary before adding AI complexity.

```text
HTTP → FastAPI controller → Pydantic validation → service → typed JSON
```

Stage 1 made `GET /health`, `GET /api/models`, and `POST /api/synthesis`
executable. Synthesis returned deterministic mock data while controllers stayed
limited to HTTP translation, validated schemas, and service calls. Automated
tests covered health, models, valid synthesis, and malformed-input `422` errors.

**Portfolio proof:** FastAPI, Pydantic, API contracts, and service separation.

## Stage 2 — First real open-weight synthesis

> “I replaced the mock implementation with locally hosted open-weight TTS
> inference without changing the API contract.”

| | |
| --- | --- |
| **Model** | Kokoro v1.0 FP32 |
| **Runtime** | ONNX Runtime on the local CPU |
| **Hosting** | Self-hosted |
| **External inference APIs** | None |
| **Current proof** | Text becomes a playable, locally served 24 kHz WAV. |

```text
POST /api/synthesis
  ↓
SynthesisService
  ↓
ModelRegistry → ModelLoader (load once)
  ↓
TTSInferenceEngine
  ↓
KokoroONNXEngine
  ↓
AudioService → generated WAV
```

Controllers still know nothing about Kokoro or ONNX. `ModelLoader` owns the
long-lived runtime session. Repeated identical requests reuse both the loaded
model and a stable request-addressed artifact.

## Stage 3 — Complete synthesis vertical slice

> “I built the first complete user-facing path from Angular input to locally
> hosted AI inference and audio playback.”

**Responsibility:** connect the product path without leaking backend technology
into the browser.

```text
Angular → FastAPI → Kokoro → Audio → Angular player
```

Stage 3 loads model and voice choices from the API, validates text, exposes a
locked loading state, maps backend and inference failures to recovery steps,
and plays the generated WAV in the browser. Angular depends only on the public
request/response contracts—never ONNX sessions, model paths, or Python classes.

![OpenVoice Lab synthesis UI](docs/images/synthesis-ui.png)

**Portfolio proof:** a visitor can use a real full-stack, self-hosted AI product,
not just inspect a backend experiment.

## Stage 4 — Instrument inference performance

> “I added model-level performance instrumentation so inference decisions can
> be based on measurements rather than intuition.”

**Responsibility:** separate model execution from model evaluation.

`MetricsCollector` measures each inference call independently from the engine.
The API and Angular now expose model load time, inference latency, exact audio
duration, RTF, process RSS memory, cold/warm state, and model variant.

Actual warm request measured on the local development machine:

| Measurement | Observed value |
| --- | ---: |
| **Model** | `kokoro-fp32` |
| **Inference** | 676.107 ms |
| **Audio duration** | 2,782.667 ms |
| **RTF** | 0.242971 |
| **Process memory** | 529.816 MB |
| **Model load** | 0.000 ms |
| **Lifecycle state** | Warm |

Mathematical check: `676.107 ÷ 2,782.667 = 0.242971`. Measurements vary by
hardware and workload; these values are recorded output, not benchmark claims.

![OpenVoice Lab inference metrics UI](docs/images/metrics-ui.png)

**Portfolio proof:** the project moves from “I can run AI” to “I can measure AI
systems.”

## Stage 5 — Model registry and inference variants

> “I designed the inference layer so multiple model configurations can be
> evaluated without changing product logic.”

**Responsibility:** make model configuration interchangeable. Keep product logic
out of model selection.

One ID represents one deployable configuration:

| Registry ID | Precision | Why it exists |
| --- | --- | --- |
| `kokoro-fp32` | FP32 | Full-precision baseline for performance and output comparison. |
| `kokoro-q8` | INT8 | Smaller quantized configuration for measuring latency and memory tradeoffs. |

```text
request.modelId
  ↓
ModelRegistry
  ├── kokoro-fp32 → FP32 artifact → cached KokoroONNXEngine
  └── kokoro-q8   → INT8 artifact → cached KokoroONNXEngine
```

There are no model-specific branches in Angular, controllers, or
`SynthesisService`. `/api/models` supplies labels, precision, voices, and stable
IDs; Angular renders that response and sends the chosen ID back. The registry
resolves the artifact and metadata. Switching precision is request data—not a
restart or code change.

Both variants will be benchmarked with the same text, voice, runtime, and
hardware. The comparison will cover load time, inference latency, RTF, process
memory, generated duration, and output quality. Stage 5 does not claim a winner;
it establishes the honest comparison boundary.

**Acceptance proof:** the real integration test synthesizes the same sentence
through both local ONNX variants in one FastAPI process, verifies playable WAVs,
confirms distinct variant metrics, and proves one load per configuration.

**Portfolio proof:** model-serving architecture with interchangeable runtime
configurations—not a one-model integration.

## Stage 6 — Reproducible benchmark system

> “I built an automated evaluation pipeline to compare inference
> configurations using identical workloads.”

**Responsibility:** turn model selection into evidence.

```text
Versioned evaluation cases
  ↓
isolated process per model
  ↓
SynthesisService → metrics → raw case results
  ↓
BenchmarkEvaluator → comparable aggregates
```

The `1.0.0` corpus covers short and medium conversation, long-form text,
numbers, dates, punctuation, questions, and unusual names or words. Every model
receives the same eight case IDs, text, and voice. Each case is retained as a
success or failure; the evaluator calculates average, median, and p95 latency,
average RTF, average and peak process memory, and failure count.

Actual run: `benchmark-2026-08-26T11-02-02-123480Z.json`, Windows 11, Python
3.13.5, 16 logical CPUs, `af_heart`, 8 identical cases per model, 0 failures.
Each model ran in a fresh process so its RSS measurement excludes the other
model. Corpus SHA-256:
`eaf6215e4cf13e670e0b3cfb56f33b6a50939a61e30a2b3296ed7c44d1d9cb98`.

| Variant | Avg latency | P95 latency | Avg RTF | Peak RSS memory |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 1,273.603 ms | 2,571.653 ms | 0.217205 | 766.098 MB |
| Q8 / INT8 | 12,865.738 ms | 25,304.652 ms | 2.151270 | 601.859 MB |

On this CPU/runtime, INT8 reduced peak RSS by 164.239 MB, or 21.4%, but average
latency was about 10.1× higher and slower than real time. I would deploy FP32
for latency-sensitive local synthesis on this machine. Q8 is defensible only
when that memory saving matters more than response time; this single-machine
result is evidence for this environment, not a universal model claim.

Run the complete benchmark with one command:

```powershell
cd backend
.\.venv\Scripts\python.exe -m app.benchmark.runner
```

Raw inputs, measurements, failures, aggregates, corpus hash, and environment
metadata are written to `backend/benchmark-results/`. Runtime result files stay
out of Git; the measured summary above remains the portfolio record.

**Portfolio proof:** a deployment decision backed by identical workloads and
recorded measurements—not intuition.

## Stage 7 — Benchmark dashboard

> “I exposed model-evaluation results through a lightweight product interface
> instead of requiring engineers to inspect raw files.”

**Responsibility:** turn engineering evidence into a product decision surface.

The Angular dashboard loads the fixed workload contract, starts an asynchronous
benchmark job, polls model-level progress, recovers the latest job in a fresh
browser session, reports pipeline failures, and renders the completed FP32/INT8
comparison. FastAPI controllers remain thin; the benchmark job service owns
background execution and the Stage 6 runner still owns evaluation.

![OpenVoice Lab benchmark dashboard](docs/images/benchmark-dashboard.png)

The dashboard intentionally exposes only metrics needed for deployment
decisions. Average latency, p95 latency, average RTF, peak process memory, and
failure count remain visible; raw case data and environment metadata stay in the
persisted benchmark result for engineering inspection.

**Portfolio proof:** engineering evaluation and product presentation of the
same evidence, connected through explicit API contracts.

## Stage 8 — Reproducible Docker deployment

> “I made the inference environment reproducible across development and
> deployment systems.”

**Responsibility:** package the working product without weakening model,
artifact, or application boundaries.

```text
Docker Compose
  ├── model-init → verified Kokoro artifacts → named volume
  ├── backend → FastAPI + ONNX Runtime → audio/result volumes
  └── frontend → Angular production build + Nginx → backend proxy
```

The Python, Node, and Nginx base images are pinned by digest. Angular installs
from `package-lock.json`; Python installs from a Linux runtime lock. A
one-shot initializer downloads the upstream FP32, INT8, and voice files,
verifies their SHA-256 checksums, and stores them outside Git in the
`model-artifacts` volume. The backend starts only after that verification
succeeds, and the frontend starts only after FastAPI passes its health check.

Kokoro model weights are Apache-2.0 licensed. The `kokoro-onnx` wrapper and
conversion repository are MIT licensed. Exact sources, filenames, and
checksums are recorded in the [artifact provenance](backend/model-artifacts/README.md).

**Portfolio proof:** AI infrastructure, reproducibility, and deployment
discipline around the same measured application.

## Stage 9 — Deterministic text sanitization

> “I traced robotic TTS output to noisy raw input and introduced a
> deterministic sanitization boundary before inference.”

**Responsibility:** remove meaningless text noise without damaging useful
English punctuation or coupling cleanup logic to Kokoro.

```text
Raw text
  ↓
TextProcessingService → TextSanitizer
  ↓
SynthesisService → inference abstraction → Kokoro
```

Sanitization is enabled by default and independently switchable in Angular.
The backend applies Unicode and whitespace cleanup, removes control and
zero-width characters, strips isolated `./`, `$`, and `%` noise, cleans
repeated punctuation, and preserves normal commas, periods, apostrophes,
questions, and grammatical hyphens. A request that contains no speakable text
after cleanup receives `422` before a model is loaded.

The response keeps `text` as submitted and exposes `normalizedText` as the
exact string used for inference and deterministic audio identity. At Stage 9,
disabling `sanitizeText` sent the validated original unchanged; Stage 10 adds
normalization as a second independent policy.

Actual warm request recorded at Stage 9 on the local development machine:

```json
{
  "text": "OpenVoice Lab ./ --- produces clean local speech for $25 ,,, today.",
  "normalizedText": "OpenVoice Lab produces clean local speech for 25 today.",
  "audioUrl": "/audio/kokoro-fp32-af_heart-79d9b4da288a625f.wav",
  "metrics": {
    "inferenceMs": 1012.396,
    "audioDurationMs": 4490.667,
    "realTimeFactor": 0.225444,
    "memoryMb": 603.75,
    "warm": true
  }
}
```

The returned file was retrieved successfully as a playable 24 kHz WAV.
Measurements are recorded evidence from this machine, not fixed performance
claims.

**Portfolio proof:** input-quality engineering behind an explicit,
replaceable preprocessing boundary—not model-specific string replacement.

## Stage 10 — Speakable English normalization

> “I preserved the meaning of technical and symbolic text by converting it
> into deterministic, speech-friendly English before inference.”

**Responsibility:** convert supported notation into speakable English without
an LLM, external normalization API, or model-specific logic.

```text
Raw text
  ↓
TextNormalizer, when enabled
  ↓
TextSanitizer, when enabled
  ↓
exact normalizedText → SynthesisService → Kokoro
```

The backend handles USD amounts, percentages, emails, HTTP(S) URLs, relative
paths, Markdown emphasis and links, inline code, common comparison operators,
snake case, and camel case. Normalization runs first so `$25` and `15%` become
`25 dollars` and `15 percent` before sanitization can remove leftover symbols.

Both controls default on and remain independent:

| Sanitize | Normalize | Result |
| --- | --- | --- |
| On | On | Expand supported meaning, then remove remaining noise. |
| On | Off | Remove noise without semantic expansion. |
| Off | On | Expand supported notation and preserve unrelated raw characters. |
| Off | Off | Send the validated original unchanged. |

Angular sends only these policy flags and displays a compact **Text sent to
model** preview when `normalizedText` differs from `text`; every rule remains in
the backend. Benchmark JSON records both strings and both flags for each case.

Genuine warm Kokoro FP32 request from the local development machine:

```json
{
  "text": "Email dev.team@example.com -- open ./docs/api-guide.md.\nThe price is $25, with a 15% discount.",
  "modelId": "kokoro-fp32",
  "voiceId": "af_heart",
  "sanitizeText": true,
  "normalizeText": true
}
```

```json
{
  "status": "ok",
  "model": "kokoro-fp32",
  "text": "Email dev.team@example.com -- open ./docs/api-guide.md.\nThe price is $25, with a 15% discount.",
  "normalizedText": "Email dev dot team at example dot com—open docs slash api guide dot M D. The price is 25 dollars, with a 15 percent discount.",
  "audioUrl": "/audio/kokoro-fp32-af_heart-20d61f2c1fcbd897.wav",
  "metrics": {
    "modelLoadMs": 0.0,
    "inferenceMs": 2108.453,
    "audioDurationMs": 10623.292,
    "realTimeFactor": 0.198475,
    "memoryMb": 829.41,
    "warm": true,
    "modelVariant": "fp32"
  }
}
```

The returned 509,962-byte file was verified as a playable mono 24 kHz WAV.
These measurements describe this machine and request; they are not fixed
performance claims.

**Known limits:** normalization is English-focused, `$` means USD, Markdown
parsing is intentionally shallow, and arbitrary programming languages or
natural-language number expansion are out of scope. Unsupported notation is
left for the independently selected sanitizer or the TTS engine.

**Portfolio proof:** deterministic input semantics, auditable inference text,
and reproducible evaluation—not opaque prompt rewriting.

## Stage 11 — Controlled SpeechT5 adaptation

> “I trained three controlled SpeechT5 variants against locked medical-speech
> workloads, retained recoverable checkpoints, and evaluated deployment tradeoffs
> with the same test set.”

**Responsibility:** turn fine-tuning into a reproducible experiment. V1 preserves
the source distribution, V2 increases medical-term exposure, and V3 uses locked
eight-row replay blocks to balance term exposure against general speech.

All three runs used the same pinned SpeechT5 revision, secure RTX 4090 class,
BF16 precision, physical batch 16, accumulation 2, effective batch 32,
`1e-5` learning rate, seed 42, and 1,000-step ceiling. Validation and recoverable
checkpoints ran every 125 optimizer steps. The dataset lock, training
configuration, checkpoints, selected models, and final artifacts were verified
by SHA-256 before the pods were terminated.

Genuine results from the locked 662-case shared-test comparison:

| Model | Training / best step | Best validation loss | Domain-term accuracy | WER | Avg inference | Avg RTF | Peak GPU | Failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **SpeechT5 pretrained** | 0 steps | — | **91.59%** | **11.10%** | **965 ms** | 0.1838 | **950 MB** | 0 |
| V1 Baseline | 1,000 | 0.441642 | 33.41% | 70.84% | 1,080 ms | 0.1567 | 1,114 MB | 0 |
| V2 Term Balance | 625 | 0.445358 | 26.20% | 72.88% | 988 ms | **0.1566** | 1,050 MB | 0 |
| V3 Replay | 1,000 | 0.444507 | 35.10% | 70.19% | 1,233 ms | 0.1819 | 1,125 MB | 0 |

The pretrained control won decisively. V3 was the strongest adapted checkpoint,
but it remained 56.49 percentage points behind pretrained term accuracy and its
WER was 59.09 points higher. The result rejects the current fine-tuning setup as
a deployment improvement and points the next iteration toward diagnosing data,
speaker-conditioning, and catastrophic-forgetting risks—not hiding the
regression behind validation loss.

The three training runs produced 24 verified checkpoints, consumed 1.9886 total
GPU hours, and recorded an estimated RunPod cost of USD 1.47. Two controller/
monitoring interruptions were retained in provenance; neither restarted or
altered a trainer. See [the completed training evidence](docs/STAGE11_TRAINING.md).

Reusable pod creation, launch, monitoring, verified checkpoint download, and
recovery commands are documented in
[Agent Training Toolkit](docs/TRAINING_AUTOMATION.md).

**Portfolio proof:** controlled ML experimentation, immutable datasets,
recoverable training, honest failure records, and evidence-based model
selection.

## Stage 12 — Interactive experiment interface

> “I turned the training run into a public engineering surface where visitors
> can inspect the evidence and test pretrained versus adapted models themselves.”

**Responsibility:** add a third top-level **Experiment** tab without changing the
Synthesis or Benchmarks products. The page reads verified Stage 11 artifacts,
shows the full pipeline and training statistics, and runs real self-hosted
SpeechT5 comparisons on CPU. The current view compares four controlled V1
update strategies—conservative full tuning, LoRA, gradual unfreezing, and
reduction factor 1—against the measured pretrained control. Its compact hiring
walkthrough follows the experiment in engineering order: controlled data and
training methods, agent-driven execution, verified checkpoints, safe pod
termination, shared metrics, the measured decision, and live model testing.
The page explicitly separates the local agent control plane from the four
independent PyTorch training processes and keeps recovery and provenance detail
one disclosure away.

![Stage 12 SpeechT5 experiment dashboard](docs/images/stage12-experiment.png)

The live lab supports locked Stage 11 fixtures and custom text with explicit
target terms. It runs two to five models sequentially against the same text and
speaker profile, exposes each WAV as soon as it is complete, transcribes it with
the pinned local Whisper evaluator, and reports exact term accuracy, WER,
inference time, audio duration, RTF, process memory, cold/warm state, and model
provenance. Sanitization and normalization remain independent backend-owned
policies. No external inference API is used.

The pretrained SpeechT5 model is the explicit control in the interface. It is
shown alongside V1A, V1B, V1C, and V1D in both the locked aggregate and every live
result. Its aggregate was produced separately on secure RTX 4090 pod
`oxnfq72nezkgft`, using the same test-manifest hash, speaker source, vocoder,
Whisper revision, and evaluation code path. All 662 cases completed with zero
failures; the evidence manifest and representative WAV passed SHA-256
verification before the pod was terminated.

The locked 662-case aggregate currently makes the decision explicit: V1C tied
pretrained domain-term accuracy at 91.59%, V1B was the fastest adapted model at
901 ms average inference and 0.1667 RTF, but pretrained retained the lowest WER
at 11.10%. V1D recorded 68 synthesis failures and is presented as a rejected
architecture change rather than hidden from the comparison.

One exercised live CPU fixture compared pretrained with V1C on `there is too
much pain when i move my arm` using target term `arm`. Both produced playable
audio, Whisper transcribed both exactly, and the job completed without failure:

| Model | Term accuracy | WER | Inference | RTF | RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| SpeechT5 pretrained | 100% | 0% | 3,072 ms | 0.8349 | 1,026 MB |
| V1C Gradual Unfreeze | 100% | 0% | 2,856 ms | 0.7693 | 2,620 MB |

This is a cold, single-sentence interaction check—not an aggregate ranking.
The locked 662-case table above remains the deployment evidence.

### Enable the CPU experiment runtime

Stage 12 is native-only for now and requires the completed Stage 11 selected
models under `artifacts/stage11/agent-runs/`. The pretrained evaluation remains
under `artifacts/stage11/full-training/pretrained/`. Those weights and generated
comparison files are intentionally excluded from Git.

```powershell
py -3.11 -m venv .runtime\stage12-venv
.\.runtime\stage12-venv\Scripts\python.exe -m pip install `
  torch==2.5.1+cpu torchaudio==2.5.1+cpu `
  --index-url https://download.pytorch.org/whl/cpu
.\.runtime\stage12-venv\Scripts\python.exe -m pip install -e "backend[experiment]"
.\.runtime\stage12-venv\Scripts\python.exe backend\scripts\provision_experiment_models.py
Push-Location backend
..\.runtime\stage12-venv\Scripts\python.exe `
  -m app.experiments.prepare_serving_profile `
  --repo-root ..
Pop-Location
.\start.ps1
```

The provisioning command uses `hf download` with immutable revisions and writes
per-model SHA-256 manifests. `start.ps1` automatically prefers the Stage 12
environment when it exists; otherwise the original Kokoro application remains
available, and locally retained Stage 11 artifacts can still back the read-only
experiment report.

**Portfolio proof:** ML experiment communication, artifact-backed reporting,
durable local jobs, progressive audio, and direct public falsifiability.

## Run from a fresh clone with Docker

Docker Desktop or Docker Engine with Compose is the only local prerequisite.
No host Python, virtual environment, Node installation, or manual model setup
is required.

```bash
git clone https://github.com/Louis-Tr/OpenVoice-Lab.git
cd OpenVoice-Lab
docker compose up
```

On the first run, Compose builds both images and downloads roughly 470 MB of
checksum-verified model artifacts into a named volume. Later starts reuse and
re-verify that volume.

Open `http://localhost:4200`. FastAPI remains directly available at
`http://localhost:8000`, including:

```bash
curl http://localhost:8000/health
# {"status":"healthy"}

curl -X POST http://localhost:8000/api/synthesis \
  -H "Content-Type: application/json" \
  -d '{"text":"OpenVoice Lab is running in Docker.","modelId":"kokoro-fp32","voiceId":"af_heart"}'
```

Stop the stack with `docker compose down`. Generated audio, benchmark results,
and models remain in named volumes so restarts do not require reprovisioning.

## Run the full stack natively

Once the one-time setup below is complete, start both servers from the repository
root:

```powershell
.\start.ps1
```

The launcher checks the virtual environment, frontend dependencies, and local
model artifacts before opening the backend and frontend server processes. Run
`.\start.ps1 -CheckOnly` to validate prerequisites without starting anything.

Backend:

```powershell
cd backend
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe scripts\download_models.py
.\.venv\Scripts\uvicorn.exe app.main:app --reload
```

Model binaries are downloaded from the upstream release, checksum-verified,
and excluded from Git. See [artifact provenance](backend/model-artifacts/README.md).

Frontend (second terminal):

```powershell
cd frontend
npm ci
npm start
```

Open `http://localhost:4200`. The development proxy keeps `/api` and generated
`/audio` requests on the same documented browser contract.

## Verified request → audio

```text
curl -X POST http://localhost:8000/api/synthesis \
  -H "Content-Type: application/json" \
  -d '{"text":"OpenVoice Lab is running locally.","modelId":"kokoro-fp32","voiceId":"af_heart","sanitizeText":true,"normalizeText":true}'
  ↓
{"status":"ok","model":"kokoro-fp32","text":"OpenVoice Lab is running locally.","normalizedText":"OpenVoice Lab is running locally.","audioUrl":"/audio/kokoro-fp32-af_heart-27561063304ff41f.wav"}
```

Result: `kokoro-fp32-af_heart-27561063304ff41f.wav`

The automated suite verifies both real model variants, playable WAVs, useful API
errors, stable output, warm reuse, one load per configuration, and deterministic
text normalization and sanitization:

```powershell
.\.venv\Scripts\pytest.exe -q
# 71 passed locally (69 portable + 2 local-artifact integrations)
```

The Angular suite covers dynamic variant discovery and selection, empty input,
the locked loading state, two independent preprocessing policies, processed-text
preview, backend unavailability, inference failure, and successful audio
delivery:

```powershell
cd frontend
npm test
# 24 passed
```

## Boundaries and roadmap

- [Architecture](docs/ARCHITECTURE.md)
- [Module responsibilities](docs/MODULES.md)
- [API contract](docs/API.md)
- [Iterative coding map](docs/ITERATIVE_CODING_MAP.md)

**Portfolio status:** this repository now demonstrates a measured, multi-variant
TTS product and evaluation system: Angular synthesis and benchmark interfaces,
FastAPI orchestration, self-hosted open-weight inference, registry-owned
lifecycle, generated audio, mathematically verified instrumentation, a
reproducible benchmark, a browser-visible deployment comparison, and a
health-checked Docker deployment with verified external model provisioning.
It also owns deterministic, user-controlled text normalization and sanitization
before inference, with the exact evaluated text retained as evidence.
The third tab adds a separately composed SpeechT5 experiment registry and CPU
runtime, keeping research evaluation out of the Kokoro product path.
