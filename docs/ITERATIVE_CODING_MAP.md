# Iterative coding map

Build one provable vertical slice at a time. Each stage must leave contracts,
ownership, and measurements clearer than it found them.

## Stage 0 — Architecture first

**Status:** complete.

Define frontend/backend boundaries, API contracts, model lifecycle ownership,
the inference port, benchmark architecture, and documentation conventions.

**Exit evidence:** repository structure, architecture docs, module ownership,
API placeholders, clean Git history, and buildable application shells.

## Stage 1 — Contracts and composition

**Status:** complete.

Wire dependency construction without loading a real model. Define application
errors, HTTP mappings, configuration validation, and contract tests.

**Exit evidence:** deterministic endpoint tests for health, model listing, valid
and invalid synthesis requests, with no inference logic in controllers.

## Stage 2 — First real open-weight synthesis

**Status:** complete.

Implement model metadata, verified artifact provisioning, load-once lifecycle,
the replaceable inference engine, Kokoro ONNX, deterministic artifact identity,
local WAV delivery, and useful domain errors.

**Exit evidence:** the sentence “OpenVoice Lab is running locally.” produces a
playable WAV; repeated warm requests reuse one model load; real and fake-engine
tests pass without any external inference API.

## Stage 3 — Complete synthesis vertical slice

**Status:** complete.

Connect Angular text input, API-provided model/voice selection, synthesis
submission, local audio delivery, and browser playback. Keep all runtime and
model implementation details behind the REST boundary.

**Exit evidence:** a fresh browser completes the real synthesis path; UI state
tests cover empty input, loading, backend unavailability, and inference failure;
the README includes a screenshot from the running product.

## Stage 4 — Metrics

**Status:** complete.

Measure model load time, inference latency, generated duration, RTF, process RSS
memory, cold/warm state, and variant identity without coupling evaluation logic
to the model adapter.

**Exit evidence:** every successful synthesis response contains independently
tested metrics, RTF matches inference time divided by audio duration, Angular
displays the values, and the README records a real local measurement.

## Stage 5 — Model registry and inference variants

**Status:** complete.

Represent FP32 and quantized Kokoro deployments as stable, data-driven registry
configurations. Let Angular discover and select them without model-specific
mapping or branching in product logic.

**Exit evidence:** the same request runs against `kokoro-fp32` and `kokoro-q8`
in one process, both produce playable audio, both report variant identity, and
each runtime loads once and serves warm requests.

## Stage 6 — Benchmark engine

**Status:** complete.

Version the sentence corpus, execute it through the same synthesis path, and
aggregate comparable results. Record case-level failures and isolate model
processes so memory remains comparable.

**Exit evidence:** reproducible benchmark output with corpus and environment
metadata, identical case IDs for FP32 and INT8, explicit failure counts, and a
README deployment decision based on genuine local measurements.

## Stage 7 — Benchmark dashboard

**Status:** complete.

Expose the reproducible benchmark through a focused Angular workflow. Keep
evaluation in the backend while the browser owns trigger, progress, recovery,
failure feedback, and comparative presentation.

**Exit evidence:** a visitor can run the real FP32/INT8 benchmark without a
terminal, observe progress, and read the completed deployment metrics in an
accessible table. The README contains a screenshot from the running product.

## Stage 8 — Packaging and operations

**Status:** complete.

Package the Angular and FastAPI application with digest-pinned base images,
automatic checksum-verified model provisioning, persistent runtime volumes,
same-origin proxying, startup dependencies, and service health checks.

**Exit evidence:** from a clean source checkout, `docker compose up` builds the
images, provisions external model files without adding them to Git, reaches a
healthy FastAPI process, serves Angular on localhost, and produces playable
speech through the containerized inference path.

## Stage 9 — Deterministic text sanitization

**Status:** complete.

Insert a backend-owned sanitizer before inference. Remove Unicode controls,
zero-width characters, isolated path fragments, symbol noise, and repeated
punctuation without damaging ordinary English phrasing. Expose a default-on,
independent Angular/API toggle and the exact processed inference text.

**Exit evidence:** synthetic rules are deterministic and idempotent, noise-only
input returns `422` before model loading, inference receives exactly
`normalizedText`, metrics exclude preprocessing, and a genuine sanitized Kokoro
request produces a playable WAV.

## Stage 10 — Speakable English normalization

**Status:** complete.

Convert meaningful currency, percentages, URLs, email addresses, paths,
Markdown, and code fragments into deterministic spoken English before the
sanitizer runs.

**Exit evidence:** independent sanitizer/normalizer toggles cover all four
combinations, original and inference text remain auditable, and technical input
produces a playable WAV without sending raw supported notation to inference.
Benchmark raw results preserve the original text, final text, and both option
states.

## Stage 11 — Controlled SpeechT5 adaptation

**Status:** complete.

Prepare three locked training schedules, fine-tune them concurrently on separate
secure RTX 4090 pods, checkpoint every 125 optimizer steps, and evaluate every
selected model against one shared 662-case medical-speech test manifest.

**Exit evidence:** the final audit verifies 24 downloaded checkpoints, dataset
and configuration hashes, selected-model hashes, run/pod provenance, shared-test
results, 1.9886 GPU hours, USD 1.47 estimated cost, no training resumptions, and
termination of all three pods. V3 reaches the highest adapted domain-term
accuracy (35.10%) but also the highest average RTF (0.1819).

## Stage 12 — Interactive experiment interface

**Status:** complete.

Add a lazy third Angular tab that presents the verified Stage 11 pipeline,
dataset schedules, configuration, loss curves, results, integrity, and incidents.
Let visitors compare pretrained SpeechT5 with any adapted variant using locked
fixtures or custom text/terms through real local CPU synthesis and ASR scoring.

**Exit evidence:** four pinned models synthesize and transcribe real WAVs on CPU;
durable comparison jobs expose progress, cancellation, progressive audio,
term accuracy, WER, performance, and provenance. Browser acceptance covers both
modes, independent text toggles, four responsive breakpoints, playable outputs,
and no console errors. Backend tests (71 locally: 69 portable plus two
local-artifact integrations), frontend tests (24), lint, and the
production build pass.

## Working rule

No stage advances on a placeholder success claim. A capability is complete only
when its implementation, test evidence, and documentation agree.

The README is cumulative: completing a new stage adds its portfolio story and
evidence without removing any earlier stage.
