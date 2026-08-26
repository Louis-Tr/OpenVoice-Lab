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

Add inference latency, RTF, process memory, cold/warm classification, and a
retention policy around the existing audio path. Define measurement conditions
before publishing data.

**Exit evidence:** verified audio metadata and repeatable metric tests.

## Stage 5 — Benchmark engine

Version the sentence corpus, execute it through the same synthesis path, and
aggregate comparable results. Add failure and cancellation policy.

**Exit evidence:** reproducible benchmark output with corpus and environment
metadata.

## Stage 6 — Packaging and operations

Add Docker packaging, artifact mounts, deployment configuration, observability,
and distinct liveness/readiness behavior.

**Exit evidence:** documented clean-machine startup and health behavior.

## Working rule

No stage advances on a placeholder success claim. A capability is complete only
when its implementation, test evidence, and documentation agree.

The README is cumulative: completing a new stage adds its portfolio story and
evidence without removing any earlier stage.
