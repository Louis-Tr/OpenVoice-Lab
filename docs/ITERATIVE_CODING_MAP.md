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

## Stage 2 — Model registry and lifecycle

Implement model metadata, artifact validation, loading policy, cache ownership,
and readiness semantics. Keep runtime objects behind backend boundaries.

**Exit evidence:** registry/lifecycle tests using fake artifacts and adapters.

## Stage 3 — First synthesis slice

Integrate Kokoro ONNX through `TTSInferenceAdapter`. Run one text/voice/model
request through `SynthesisService` and the audio service.

**Exit evidence:** reproducible audio from a documented model artifact, with the
adapter replaceable in tests.

## Stage 4 — Audio and metrics

Add encoding, duration, artifact retention, latency, RTF, process memory, and
cold/warm classification. Define measurement conditions before publishing data.

**Exit evidence:** verified audio metadata and repeatable metric tests.

## Stage 5 — Benchmark engine

Version the sentence corpus, execute it through the same synthesis path, and
aggregate comparable results. Add failure and cancellation policy.

**Exit evidence:** reproducible benchmark output with corpus and environment
metadata.

## Stage 6 — Angular workflow

Connect model selection, synthesis, playback, metrics, benchmark execution, and
clear loading/error states strictly through REST contracts.

**Exit evidence:** end-to-end user flow with no inference details in Angular.

## Stage 7 — Packaging and operations

Add Docker packaging, artifact mounts, deployment configuration, observability,
and distinct liveness/readiness behavior.

**Exit evidence:** documented clean-machine startup and health behavior.

## Working rule

No stage advances on a placeholder success claim. A capability is complete only
when its implementation, test evidence, and documentation agree.
