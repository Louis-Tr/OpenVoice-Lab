# Stage 11 full-training evidence

This document records the frozen contract and the verified outcome of the three
completed SpeechT5 experiments. Claims below are parsed from the retained local
artifacts and final audit.

## Experiment matrix

| Run | Training schedule | Purpose |
| --- | --- | --- |
| `v1-baseline` | Uniform exposure | Establish the adapted-model baseline. |
| `v2-term-balance` | Term-balanced exposure | Test whether stronger domain-term exposure improves pronunciation. |
| `v3-replay` | Locked 50/50 term-balanced and general replay blocks | Test domain improvement while limiting general-speech regression. |

All runs use the same pinned SpeechT5 model, seed, optimizer settings, batch configuration, stopping rules, validation set, and test set. The dataset lock records the exact manifest hashes. The V3 schedule is shuffled in complete eight-row blocks; individual-row shuffling is prohibited because it would destroy the intended replay mix.

## Final training configuration

| Setting | Value |
| --- | ---: |
| GPU | Secure RTX 4090, one GPU per run |
| Precision | BF16 |
| Physical batch | 16 |
| Gradient accumulation | 2 |
| Effective batch | 32 |
| Maximum optimizer steps | 1,000 |
| Approximate epochs | 6.024 |
| Learning rate | `1e-5`, linear decay |
| Warmup | 100 steps |
| Gradient clipping | `1.0` |
| Gradient checkpointing | Enabled |
| Validation cadence | Every 125 optimizer steps |
| Early stopping | Three validation checks without an `eval_loss` improvement of at least `0.001` |
| Best model | Lowest validation loss |

The step cap is authoritative. `nominal_epochs: 6` communicates the intended training duration; the 1,000-step cap is approximately 6.03 epochs because each 5,304-row schedule requires 166 optimizer steps at effective batch 32.

## Checkpoint contract

Each validation boundary creates a recoverable checkpoint every 125 optimizer steps. Only the latest two recovery checkpoints rotate. Checkpoints at steps **250, 500, 750, and 1,000** are durable 25%, 50%, 75%, and 100% experiment milestones and must not be rotated away.

Every checkpoint must contain model weights, optimizer state, scheduler state, random-number-generator state, trainer state, configuration hash, dataset-lock hash, global step, validation metrics, and a SHA-256 file manifest. The selected best checkpoint and a final inference export are retained separately. A resumed run uses the latest checkpoint whose hashes and state files validate.

Early stopping can make later milestones inapplicable. That is expected: the best checkpoint and the last valid recovery checkpoint become the final experiment evidence, and the run record must state that convergence stopped the run.

## Launch gate override

The earlier secure RTX 4090 probe passed at physical batch 4, accumulation 2. It does **not** validate the final physical batch 16 decision. The repository owner explicitly waived the batch-16 stability probe before the full runs. This waiver is recorded as `user_authorized_skip` in the frozen configuration and must appear in every run's provenance. Dataset locks, manifest hashes, finite-value checks, disk checks, and checkpoint-integrity checks remain mandatory.

Validate the frozen configuration and dataset hashes locally:

```powershell
python -m training.full_training.preflight
```

The machine-readable report is written to
`artifacts/stage11/full-training/preflight.json`.

## Verified completion

Run ID: `stage11-speecht5-full-20260827-043116`

| Variant | Pod | Final step | Best step | Best eval loss | Term accuracy | WER | Avg RTF |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 Baseline | `ja4erg2qtyd31z` | 1,000 | 1,000 | 0.441642 | 33.41% | 70.84% | 0.1567 |
| V2 Term Balance | `efpo8hfkkg6yd2` | 1,000 | 625 | 0.445358 | 26.20% | 72.88% | 0.1566 |
| V3 Replay | `gumimdul2kqefm` | 1,000 | 1,000 | 0.444507 | 35.10% | 70.19% | 0.1819 |

All variants completed the 1,000-step ceiling; early stopping did not fire.
Every 125-step checkpoint was downloaded and verified, producing 24 valid
checkpoints total. The shared test set contained 662 cases. The three pods used
1.9886 total GPU hours at an estimated USD 1.47 and were terminated only after
the required artifacts were stored and audited locally.

V3 produced the strongest adapted domain-term accuracy and lowest WER, but at a
higher runtime cost. V1 produced the lowest validation loss. V2's term-balanced
schedule did not improve the shared-test pronunciation proxy, which is a useful
negative result rather than evidence to hide.

Two operational incidents remain in provenance: the local controller was
restarted with explicit UTF-8 replacement decoding, and the V3 monitor briefly
lost SSH connectivity around step 730. The trainers continued; neither incident
caused a training restart or changed the locked data/configuration.

The authoritative evidence is under `artifacts/stage11/full-training/`:
`final-audit.json`, per-variant `run_provenance.json`, `training_metadata.json`,
`evaluation.json`, `run_artifact_manifest.json`, selected models, and checkpoint
inventories. These large runtime artifacts are intentionally excluded from Git.
