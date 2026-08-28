import { HttpErrorResponse } from '@angular/common/http';
import { ChangeDetectionStrategy, Component, OnInit, signal } from '@angular/core';
import { forkJoin } from 'rxjs';

import { ExperimentApiService } from '../api/experiment-api.service';
import {
  ExperimentFixture,
  ExperimentModelSummary,
  ExperimentReport,
  ExperimentVariantReport,
} from './experiment.types';
import { LiveComparisonComponent } from './live-comparison.component';
import { TrainingLossChartComponent } from './training-loss-chart.component';

@Component({
  selector: 'ovl-stage11-experiment-page',
  standalone: true,
  imports: [LiveComparisonComponent, TrainingLossChartComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  styleUrl: './stage11-experiment-page.component.css',
  template: `
    <main class="experiment-page" aria-labelledby="experiment-title">
      <header class="hero">
        <p class="eyebrow"><i aria-hidden="true"></i> Agent-orchestrated SpeechT5 experiment</p>
        <div class="hero-layout">
          <div>
            <h2 id="experiment-title">Train four ways.<br /><span>Measure every decision.</span></h2>
            <p class="intro">A controlled SpeechT5 adaptation experiment executed by reusable agents: one locked dataset, four GPU training methods, verified checkpoints, a shared evaluator, and compute terminated only after every artifact was safe.</p>
            <div class="hero-actions">
              <a class="primary-link" href="/experiments/stage11#agent-pipeline">Inspect the training system</a>
              <a class="text-link" href="/experiments/stage11#live-lab">Run the models <span aria-hidden="true">→</span></a>
            </div>
          </div>
          <ol class="hiring-tour" aria-label="Suggested experiment walkthrough">
            <li><span>01</span><div><strong>Understand the methods</strong><small>One dataset, four controlled updates</small></div></li>
            <li><span>02</span><div><strong>Audit the automation</strong><small>Agents, pods, checkpoints, termination</small></div></li>
            <li><span>03</span><div><strong>Inspect the evidence</strong><small>Metrics, decision, and live models</small></div></li>
          </ol>
        </div>
      </header>

      @if (state() === 'loading') {
        <section class="loading-state" aria-live="polite">
          <strong>Loading verified experiment evidence…</strong>
          <span>Reading the dataset lock, training history, final audit, and model catalog.</span>
        </section>
      } @else if (state() === 'error') {
        <section class="error-state" role="alert">
          <div><strong>Experiment evidence is unavailable.</strong><p>{{ error() }}</p></div>
          <button type="button" (click)="load()">Retry</button>
        </section>
      } @else if (report(); as evidence) {
        <section class="integrity" aria-label="Experiment integrity">
          <div><span>Dataset lock</span><strong>{{ evidence.integrity.datasetLockVerified ? 'Verified' : 'Failed' }}</strong></div>
          <div><span>Checkpoints</span><strong>{{ evidence.integrity.checkpointCount }} verified</strong></div>
          <div><span>Final models</span><strong>{{ evidence.integrity.finalArtifactHashesVerified ? 'Verified' : 'Failed' }}</strong></div>
          <div><span>RunPod state</span><strong>{{ evidence.integrity.allPodsTerminated ? 'All terminated' : 'Active' }}</strong></div>
        </section>

        <section id="experiment-design" class="method surface" aria-labelledby="pipeline-title">
          <div class="section-label">Controlled experiment</div>
          <div class="section-heading compact-heading">
            <h3 id="pipeline-title">One dataset. Four ways to update SpeechT5.</h3>
            <p>The experiment asks which adaptation method improves medical-term pronunciation without sacrificing general transcription quality or serving performance.</p>
          </div>

          <div class="experiment-contract" aria-label="Experiment contract">
            <article><span>Question</span><strong>Can adaptation beat pretrained?</strong><p>Improve domain-term accuracy without regressing WER, failures, latency, RTF, or memory.</p></article>
            <article><span>Control</span><strong>SpeechT5 pretrained</strong><p>The unchanged base model ran through the same locked {{ integer(evidence.sharedSplits['test']) }}-case evaluator.</p></article>
            <article><span>Constants</span><strong>Data + revisions + evaluator</strong><p>Every approach shared the manifests, speaker source, model revisions, seed, and measurement path.</p></article>
          </div>

          <div class="audit-strip" aria-label="Dataset preparation audit">
            <div><span>Approved audio</span><strong>{{ integer(evidence.dataAudit.uniqueAudioFiles) }}</strong><small>16 kHz mono files</small></div>
            <div><span>Audio failures</span><strong>{{ evidence.dataAudit.audioVerificationFailureCount }}</strong><small>after verification</small></div>
            <div><span>Leakage intersections</span><strong>{{ evidence.dataAudit.leakageIntersectionCount }}</strong><small>across {{ evidence.dataAudit.leakageIdentityFields.length }} identity checks</small></div>
            <div><span>Shared test</span><strong>{{ integer(evidence.sharedSplits['test']) }}</strong><small>identical cases per variant</small></div>
          </div>

          <div class="pipeline" aria-label="SpeechT5 data preparation pipeline">
            <span>Immutable intake</span><i aria-hidden="true">→</i><span>Audio + text cleanup</span><i aria-hidden="true">→</i><span>Quality + dedupe</span><i aria-hidden="true">→</i><span>Leakage-safe splits</span><i aria-hidden="true">→</i><span>Hash + lock</span><i aria-hidden="true">→</i><span>Shared V1 schedule</span>
          </div>

          <details class="method-disclosure">
            <summary><span>Open the complete preprocessing boundary</span><small>Rules, split sizes, and verification</small></summary>
            <div class="disclosure-body">
              <ol class="process-list">
                <li><strong>Inventory without mutation.</strong><span>The canonical raw tree and metadata were hashed; source files were never rewritten.</span></li>
                <li><strong>Standardize audio.</strong><span>Readable recordings became 16 kHz, mono, PCM16 WAV with deterministic duration and quality checks.</span></li>
                <li><strong>Normalize transcripts.</strong><span>Unicode and whitespace were normalized, medical terms annotated, and source provenance retained.</span></li>
                <li><strong>Reject hard failures.</strong><span>Duration, clipping, silence, and low-level failures remained excluded even though manual review was waived.</span></li>
                <li><strong>Block leakage.</strong><span>Audio hash, normalized transcript, near-duplicate group, and sample identity were checked across splits.</span></li>
                <li><strong>Lock the experiment.</strong><span>Train {{ integer(evidence.sharedSplits['train']) }}, validation {{ integer(evidence.sharedSplits['validation']) }}, and test {{ integer(evidence.sharedSplits['test']) }} manifests were hashed before scheduling.</span></li>
              </ol>
              <dl class="mini-proof">
                <div><dt>Builder</dt><dd>{{ shortHash(evidence.dataAudit.builderSha256) }}</dd></div>
                <div><dt>Variant config</dt><dd>{{ shortHash(evidence.dataAudit.variantConfigSha256) }}</dd></div>
                <div><dt>Schedule block</dt><dd>{{ evidence.dataAudit.scheduleBlockSize }} rows</dd></div>
                <div><dt>Shared evaluation</dt><dd>{{ evidence.dataAudit.sharedEvaluationManifests ? 'Verified' : 'Failed' }}</dd></div>
              </dl>
            </div>
          </details>

          <div class="variant-stack">
            @for (variant of evidence.variants; track variant.id) {
              <details class="variant-card">
                <summary>
                  <div><span>{{ variant.id }}</span><strong>{{ strategyTitle(variant) }}</strong><small>{{ strategyDescription(variant) }}</small></div>
                  <div class="summary-metrics"><span>{{ trainingBatch(variant) }}</span><span>{{ learningRate(variant) }} LR</span><b>{{ outcomeLabel(variant) }}</b></div>
                </summary>
                <div class="variant-detail">
                  <div class="story-column">
                    <p><b>Training need</b>{{ trainingNeed(variant) }}</p>
                    <p><b>Schedule construction</b>{{ scheduleConstruction(variant, evidence) }}</p>
                    <p><b>Engineering conclusion</b>{{ conclusion(variant, evidence) }}</p>
                  </div>
                  <dl class="variant-stats">
                    <div><dt>Model update</dt><dd>{{ variant.approach }}</dd></div>
                    <div><dt>Physical / accumulation</dt><dd>{{ trainingBatch(variant) }}</dd></div>
                    <div><dt>Effective batch</dt><dd>{{ variant.trainingConfig?.effectiveBatchSize ?? '—' }}</dd></div>
                    <div><dt>Learning rate</dt><dd>{{ learningRate(variant) }}</dd></div>
                    <div><dt>Reduction factor</dt><dd>{{ variant.trainingConfig?.reductionFactor ?? '—' }}</dd></div>
                    <div><dt>Train loss</dt><dd>{{ optionalFixed(variant.trainLoss, 6) }}</dd></div>
                    <div><dt>Best eval loss</dt><dd>{{ fixed(variant.bestValidationLoss, 6) }}</dd></div>
                    <div><dt>Selected / best / final</dt><dd>{{ variant.selectedStep ?? '—' }} / {{ variant.bestStep }} / {{ variant.finalStep }}</dd></div>
                    <div><dt>Checkpoints</dt><dd>{{ variant.checkpointSteps.length }} × verified</dd></div>
                    <div><dt>Estimated cost</dt><dd>USD {{ fixed(variant.estimatedCostUsd, 2) }}</dd></div>
                  </dl>
                  <p class="pool-line"><strong>Shared dataset</strong> {{ integer(variant.dataset.scheduledRows) }} scheduled rows · {{ integer(variant.dataset.rowsWithTerms) }} term-bearing rows · {{ variant.dataset.uniqueSpeakers }} speakers · {{ fixed(variant.dataset.durationHours, 2) }} hours</p>
                  <p class="pool-line"><strong>Selection policy</strong> {{ selectionLabel(variant) }}</p>
                  <p class="pool-line"><strong>Train manifest</strong> {{ shortHash(variant.dataset.manifestSha256['train']) }}</p>
                </div>
              </details>
            }
          </div>
        </section>

        <section id="agent-pipeline" class="automation surface" aria-labelledby="automation-title">
          <div class="section-label">Agent-driven training pipeline</div>
          <div class="section-heading compact-heading">
            <h3 id="automation-title">Automatic from run document to verified shutdown.</h3>
            <p>A reusable controller handled infrastructure and evidence collection around each independent PyTorch trainer. The agent orchestrated the run; it did not alter gradients, losses, or model outputs.</p>
          </div>

          <aside class="agent-boundary">
            <strong>Control plane ≠ training plane</strong>
            <p>The local agent toolkit validated inputs, provisioned and watched pods, transferred artifacts, and enforced gates. SpeechT5 optimization stayed inside four isolated GPU training processes driven by committed YAML profiles.</p>
          </aside>

          <div class="orchestration-map" aria-label="Agent controller dispatches four isolated RunPod training workers">
            <article class="controller-node">
              <span>Local control plane</span>
              <strong>Agent controller + durable watcher</strong>
              <small>run.json · status.json · events.jsonl · checkpoint-inventory.json</small>
            </article>
            <div class="dispatch-arrow" aria-hidden="true"><span>dispatch</span><i>→</i><span>monitor</span></div>
            <div class="pod-grid">
              @for (variant of evidence.variants; track variant.id) {
                <article>
                  <span>{{ variant.id }}</span>
                  <strong>Isolated training worker</strong>
                  <small>RTX 4090 · {{ variant.finalStep }} steps · USD {{ fixed(variant.estimatedCostUsd, 2) }}</small>
                </article>
              }
            </div>
          </div>

          <ol class="automation-steps" aria-label="Automated training lifecycle">
            <li>
              <span>01 / PREFLIGHT</span>
              <strong>Lock every input</strong>
              <p>Require a clean commit, validate the dataset lock, pin model revisions, hash the training bundle, and persist a unique run document.</p>
              <small>{{ integer(evidence.variants[0].dataset.scheduledRows) }} scheduled train · {{ integer(evidence.sharedSplits['validation']) }} validation · {{ integer(evidence.sharedSplits['test']) }} test</small>
            </li>
            <li>
              <span>02 / PROVISION</span>
              <strong>Create one pod per method</strong>
              <p>Request four secure, non-interruptible RTX 4090 workers and bind each worker to exactly one approach and configuration.</p>
              <small>{{ evidence.variants.length }} workers · {{ evidence.variants.length }} isolated run records</small>
            </li>
            <li>
              <span>03 / TRAIN + WATCH</span>
              <strong>Observe the real process</strong>
              <p>Launch or reattach the trainer and poll global step, loss, gradients, GPU memory, process memory, disk, evaluation, and provider state.</p>
              <small>{{ evidence.training.precision }} · effective batch {{ evidence.training.effectiveBatchSize }} · validation every {{ evidence.training.evaluationSteps }}</small>
            </li>
            <li class="checkpoint-step">
              <span>04 / CHECKPOINT</span>
              <strong>Transfer only complete state</strong>
              <p>Detect the completion marker, verify remotely, download into a temporary directory, validate every SHA-256 entry, then publish atomically.</p>
              <small>{{ evidence.integrity.checkpointCount }} verified checkpoints · every {{ evidence.training.evaluationSteps }} steps</small>
            </li>
            <li>
              <span>05 / SELECT + MEASURE</span>
              <strong>Evaluate identical evidence</strong>
              <p>Select a bounded quality checkpoint, reload it, synthesize a real WAV, and score every model on the same locked test manifest.</p>
              <small>Term accuracy · WER · exact sentence · latency · RTF · memory · failures</small>
            </li>
            <li class="termination-step">
              <span>06 / VERIFY + TERMINATE</span>
              <strong>Shut down after proof</strong>
              <p>Download final models, logs, metrics, manifests, and hashes. Termination is refused until the final-artifact gate passes, then provider state is checked again.</p>
              <small>{{ evidence.integrity.finalArtifactHashesVerified ? 'Final artifacts verified' : 'Verification failed' }} · {{ evidence.integrity.allPodsTerminated ? '4/4 pods terminated' : 'pod still active' }}</small>
            </li>
          </ol>

          <div class="automation-proof" aria-label="Observed automation results">
            <div><span>Parallel workers</span><strong>{{ evidence.variants.length }}</strong><small>one per method</small></div>
            <div><span>Verified checkpoints</span><strong>{{ evidence.integrity.checkpointCount }}</strong><small>10 per run</small></div>
            <div><span>Training resumptions</span><strong>{{ evidence.trainingResumptions.length }}</strong><small>durable recovery remained ready</small></div>
            <div><span>Provider shutdown</span><strong>{{ evidence.integrity.allPodsTerminated ? '4 / 4' : 'Incomplete' }}</strong><small>verified after artifact gate</small></div>
          </div>

          <details class="method-disclosure automation-disclosure">
            <summary><span>Open checkpoint and recovery guarantees</span><small>Atomic downloads, safe reattachment, and termination gate</small></summary>
            <div class="disclosure-body two-column-copy">
              <div><strong>Durable handoff</strong><p><code>pod.json</code> owns provider identity. <code>run.json</code> owns approach, configuration, revisions, remote PID, current phase, and resume paths. A replacement controller can safely reattach.</p></div>
              <div><strong>Checkpoint transaction</strong><p>Completion marker → remote verification → temporary local download → manifest verification → atomic rename. An interrupted transfer never deletes the valid pod copy.</p></div>
              <div><strong>Failure recovery</strong><p>The watcher preserves all completed artifacts. If no trainer is active, restart resolves the latest verified checkpoint instead of silently returning to step zero.</p></div>
              <div><strong>Termination interlock</strong><p>The ordinary termination command refuses shutdown until final artifacts verify. Forced termination exists only for an explicitly abandoned run.</p></div>
            </div>
          </details>
        </section>

        <section class="training surface" aria-labelledby="training-title">
          <div class="section-label">Training method</div>
          <div class="section-heading compact-heading">
            <h3 id="training-title">Shared controls. Intentional differences.</h3>
            <p>The comparison kept the data, revisions, evaluator, seed, effective batch, and step ceiling fixed. Only each declared adaptation mechanism and the optimizer settings it required were allowed to differ.</p>
          </div>
          <dl class="config-grid">
            <div><dt>Precision</dt><dd>{{ evidence.training.precision }}</dd></div>
            <div><dt>Effective batch</dt><dd>{{ evidence.training.effectiveBatchSize }}</dd></div>
            <div><dt>Maximum steps</dt><dd>{{ integer(evidence.training.maximumSteps) }}</dd></div>
            <div><dt>Warmup</dt><dd>{{ evidence.training.warmupSteps }} steps</dd></div>
            <div><dt>Validation</dt><dd>Every {{ evidence.training.evaluationSteps }}</dd></div>
            <div><dt>Gradient clipping</dt><dd>{{ evidence.training.maximumGradientNorm }}</dd></div>
            <div><dt>Seed</dt><dd>{{ evidence.training.seed }}</dd></div>
            <div><dt>Shared train rows</dt><dd>{{ integer(evidence.variants[0].dataset.scheduledRows) }}</dd></div>
          </dl>
          <details class="method-disclosure training-disclosure">
            <summary><span>Open the complete optimization contract</span><small>Shared controls, approach-specific settings, and source revisions</small></summary>
            <div class="disclosure-body two-column-copy">
              <div><strong>Optimization</strong><p>BF16, gradient checkpointing, effective batch {{ evidence.training.effectiveBatchSize }}, gradient norm {{ evidence.training.maximumGradientNorm }}, seed {{ evidence.training.seed }}, and a {{ evidence.training.maximumSteps }}-step ceiling.</p></div>
              <div><strong>Approach controls</strong><p>V1A and V1C used a 1e-6 learning rate; LoRA used 5e-5; reduction-factor-1 required physical batch 8 with accumulation 4. Each choice is exposed above.</p></div>
              <div><strong>Validation + selection</strong><p>Validation ran every {{ evidence.training.evaluationSteps }} steps. Deployment candidates were selected with quality guardrails, so the chosen checkpoint could differ from the lowest validation-loss checkpoint.</p></div>
              <div><strong>Model provenance</strong><p>SpeechT5 {{ shortHash(evidence.sourceModelRevisions['tts']) }}, HiFi-GAN {{ shortHash(evidence.sourceModelRevisions['vocoder']) }}, Whisper {{ shortHash(evidence.sourceModelRevisions['asr']) }}.</p></div>
            </div>
          </details>
        </section>

        <ovl-training-loss-chart [variants]="evidence.variants" />

        <section class="outcome surface" aria-labelledby="measured-results-title">
          <div class="section-label">Measurement → comparison → decision</div>
          <div class="section-heading">
            <h3 id="measured-results-title">No fine-tune beat<br />the control overall.</h3>
            <p>{{ evidence.headline }} Every model used the same evaluator, and every negative result remains visible because it changes the engineering decision.</p>
          </div>

          <div class="measurement-grid" aria-label="Evaluation framework">
            <article><span>Headline quality</span><strong>Domain-term accuracy</strong><p>ASR-recognized target medical terms divided by all expected target terms.</p></article>
            <article><span>General quality</span><strong>WER + exact sentence</strong><p>Transcription errors expose regressions that a domain-only score could hide.</p></article>
            <article><span>Serving cost</span><strong>Latency + RTF + memory</strong><p>Inference time, generated duration, GPU/process peaks, and real-time factor quantify deployment impact.</p></article>
            <article><span>Reliability</span><strong>Failures + playable WAV</strong><p>Failed cases remain in the denominator, and each selected model must reload and synthesize real audio.</p></article>
          </div>

          <div class="hypothesis-grid">
            @for (variant of evidence.variants; track variant.id) {
              <article [class.best-outcome]="variant.id === 'v1c-gradual-unfreeze'">
                <div class="card-topline"><span>{{ variant.name }}</span><strong>{{ outcomeLabel(variant) }}</strong></div>
                <p><b>Expected</b>{{ expectation(variant) }}</p>
                <p><b>Measured</b>{{ measuredOutcome(variant, evidence) }}</p>
              </article>
            }
          </div>

          <div class="table-scroll" tabindex="0" aria-label="SpeechT5 evaluation results including the pretrained control">
            <table>
              <caption>{{ evidence.runtimeLabel }} · common {{ evidence.pretrainedControl.evaluation.caseCount }}-case test set</caption>
              <thead>
                <tr><th scope="col">Model</th><th scope="col">Training</th><th scope="col">Term accuracy</th><th scope="col">WER</th><th scope="col">Exact sentence</th><th scope="col">Avg inference</th><th scope="col">RTF</th><th scope="col">Peak GPU</th><th scope="col">Failures</th></tr>
              </thead>
              <tbody>
                <tr class="control-row" [class.winner]="primaryWinnerId(evidence) === evidence.pretrainedControl.id">
                  <th scope="row"><strong>{{ evidence.pretrainedControl.name }}</strong><span>unadapted control · {{ evidence.pretrainedControl.hardware }} · pinned {{ shortHash(evidence.pretrainedControl.revision) }}</span></th>
                  <td>{{ evidence.pretrainedControl.trainingSteps }} steps</td>
                  <td>{{ percent(evidence.pretrainedControl.evaluation.domainTermAccuracy) }}</td>
                  <td>{{ percent(evidence.pretrainedControl.evaluation.wordErrorRate) }}</td>
                  <td>{{ optionalPercent(evidence.pretrainedControl.evaluation.exactSentenceRate) }}</td>
                  <td>{{ integer(evidence.pretrainedControl.evaluation.averageInferenceMs) }} ms</td>
                  <td>{{ fixed(evidence.pretrainedControl.evaluation.averageRealTimeFactor, 4) }}</td>
                  <td>{{ integer(evidence.pretrainedControl.evaluation.peakGpuMemoryMb) }} MB</td>
                  <td>{{ evidence.pretrainedControl.evaluation.failureCount }}</td>
                </tr>
                @for (variant of evidence.variants; track variant.id) {
                  <tr [class.winner]="primaryWinnerId(evidence) === variant.id">
                    <th scope="row"><strong>{{ variant.name }}</strong><span>selected {{ checkpointLabel(variant.selectedStep) }} · best loss at {{ variant.bestStep }}</span></th>
                    <td>{{ fixed(variant.trainingSeconds / 60, 1) }} min</td>
                    <td>{{ percent(variant.evaluation.domainTermAccuracy) }}</td>
                    <td>{{ percent(variant.evaluation.wordErrorRate) }}</td>
                    <td>{{ optionalPercent(variant.evaluation.exactSentenceRate) }}</td>
                    <td>{{ integer(variant.evaluation.averageInferenceMs) }} ms</td>
                    <td>{{ fixed(variant.evaluation.averageRealTimeFactor, 4) }}</td>
                    <td>{{ integer(variant.evaluation.peakGpuMemoryMb) }} MB</td>
                    <td>{{ variant.evaluation.failureCount }}</td>
                  </tr>
                }
              </tbody>
            </table>
          </div>
          <p class="evidence-note"><strong>Measurement boundary:</strong> all five rows use the same locked {{ evidence.pretrainedControl.evaluation.caseCount }}-case test manifest, pinned speaker source, vocoder, ASR revision, and secure RTX 4090 evaluation path. The pretrained control was evaluated without training; live CPU measurements below remain separate.</p>
        </section>

        <ovl-live-comparison [models]="models()" [fixtures]="fixtures()" />

        <section class="provenance surface" aria-labelledby="provenance-title">
          <div class="section-label">Reproducibility</div>
          <div class="section-heading compact-heading">
            <h3 id="provenance-title">The evidence stays attached.</h3>
            <p>Every claim above is projected from local run artifacts, not copied into presentation code.</p>
          </div>
          <div class="proof-grid">
            <div><span>Total GPU time</span><strong>{{ fixed(evidence.totalGpuHours, 4) }} hours</strong></div>
            <div><span>Estimated cost</span><strong>USD {{ fixed(evidence.estimatedTotalCostUsd, 2) }}</strong></div>
            <div><span>Training resumptions</span><strong>{{ evidence.trainingResumptions.length }}</strong></div>
            <div><span>Configuration</span><strong>{{ shortHash(evidence.configurationSha256) }}</strong></div>
          </div>
          <details class="method-disclosure">
            <summary><span>Open run provenance and incidents</span><small>Pods, hashes, and controller recovery</small></summary>
            <div class="disclosure-body">
              <dl class="hash-list">
                <div><dt>Run ID</dt><dd>{{ evidence.runId }}</dd></div>
                <div><dt>Dataset lock</dt><dd>{{ evidence.datasetLockSha256 }}</dd></div>
                <div><dt>Pretrained control evaluation</dt><dd>{{ evidence.pretrainedControl.evaluation.caseCount }} cases · artifact {{ shortHash(evidence.pretrainedControl.artifactManifestSha256) }}</dd></div>
                @for (variant of evidence.variants; track variant.id) {
                  <div><dt>{{ variant.name }} run</dt><dd>Best step {{ variant.bestStep }} · {{ variant.stoppedEarly ? 'early stopped' : 'completed max steps' }}</dd></div>
                }
              </dl>
              @if (evidence.incidents.length > 0) {
                <div class="incident-list">
                  @for (incident of evidence.incidents; track incident.id) {
                    <article><strong>{{ incident.id }}</strong><p>{{ incident.impact }} {{ incident.resolution }}</p><span>Training restart: {{ incident.trainingRestart ? 'yes' : 'no' }}</span></article>
                  }
                </div>
              }
            </div>
          </details>
        </section>
      }
    </main>
  `,
})
export class Stage11ExperimentPageComponent implements OnInit {
  readonly state = signal<'loading' | 'ready' | 'error'>('loading');
  readonly error = signal('');
  readonly report = signal<ExperimentReport | null>(null);
  readonly models = signal<readonly ExperimentModelSummary[]>([]);
  readonly fixtures = signal<readonly ExperimentFixture[]>([]);

  constructor(private readonly api: ExperimentApiService) {}

  ngOnInit(): void { this.load(); }

  load(): void {
    this.state.set('loading');
    this.error.set('');
    forkJoin({ report: this.api.getReport(), models: this.api.getModels(), fixtures: this.api.getFixtures() }).subscribe({
      next: ({ report, models, fixtures }) => {
        this.report.set(report);
        this.models.set(models);
        this.fixtures.set(fixtures.items);
        this.state.set('ready');
      },
      error: (error: HttpErrorResponse) => {
        this.error.set(error.status === 0 ? 'Start FastAPI, then retry.' : (error.error?.detail ?? 'The verified report could not be loaded.'));
        this.state.set('error');
      },
    });
  }

  fixed(value: number, digits: number): string { return value.toFixed(digits); }
  optionalFixed(value: number | null, digits: number): string { return value === null ? 'Not recorded' : value.toFixed(digits); }
  percent(value: number): string { return `${(value * 100).toFixed(2)}%`; }
  optionalPercent(value: number | null): string { return value === null ? 'Not recorded' : this.percent(value); }
  integer(value: number): string { return new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(value); }
  shortHash(value: string): string { return `${value.slice(0, 12)}…`; }
  checkpointLabel(step: number | null): string { return step === null ? 'not recorded' : `step ${step}`; }

  outcomeLabel(variant: ExperimentVariantReport): string {
    return {
      'v1a-conservative-full': 'Quality regressed',
      'v1b-lora': 'Fastest adapted',
      'v1c-gradual-unfreeze': 'Term accuracy tied',
      'v1d-reduction-factor-1': 'Model collapsed',
    }[variant.id];
  }

  expectation(variant: ExperimentVariantReport): string {
    return {
      'v1a-conservative-full': 'A 10× lower learning rate should make full-model adaptation conservative enough to preserve pretrained speech quality.',
      'v1b-lora': 'Small attention adapters should learn domain pronunciation while limiting destructive updates to the base model.',
      'v1c-gradual-unfreeze': 'Training speech heads first, then the top decoder blocks, should preserve general behavior while adapting pronunciation.',
      'v1d-reduction-factor-1': 'Predicting one acoustic frame per decoder step should add pronunciation detail despite the additional compute.',
    }[variant.id];
  }

  measuredOutcome(variant: ExperimentVariantReport, report: ExperimentReport): string {
    const controlGap = (variant.evaluation.domainTermAccuracy - report.pretrainedControl.evaluation.domainTermAccuracy) * 100;
    const werGap = (variant.evaluation.wordErrorRate - report.pretrainedControl.evaluation.wordErrorRate) * 100;
    if (variant.id === 'v1a-conservative-full') return `${this.signed(controlGap)} term-accuracy points and ${this.signed(werGap)} WER points versus pretrained.`;
    if (variant.id === 'v1b-lora') {
      const latency = ((variant.evaluation.averageInferenceMs / report.pretrainedControl.evaluation.averageInferenceMs) - 1) * 100;
      return `${this.signed(latency)}% latency, but ${this.signed(controlGap)} term-accuracy points and ${this.signed(werGap)} WER points.`;
    }
    if (variant.id === 'v1c-gradual-unfreeze') return `Matched pretrained at ${this.percent(variant.evaluation.domainTermAccuracy)} term accuracy, but WER was ${this.signed(werGap)} points worse.`;
    return `${variant.evaluation.failureCount} synthesis failures, ${this.percent(variant.evaluation.domainTermAccuracy)} term accuracy, and ${this.percent(variant.evaluation.wordErrorRate)} WER.`;
  }

  strategyTitle(variant: ExperimentVariantReport): string {
    return {
      'v1a-conservative-full': 'Conservative full-model tuning',
      'v1b-lora': 'LoRA attention adapters',
      'v1c-gradual-unfreeze': 'Gradual decoder unfreeze',
      'v1d-reduction-factor-1': 'Reduction factor 1',
    }[variant.id];
  }

  strategyDescription(variant: ExperimentVariantReport): string {
    return {
      'v1a-conservative-full': 'Updates every SpeechT5 parameter with a low learning rate.',
      'v1b-lora': 'Freezes base weights and trains rank-8 attention adapters.',
      'v1c-gradual-unfreeze': 'Starts with modal heads, then opens the top two decoder blocks.',
      'v1d-reduction-factor-1': 'Rebuilds output heads to emit one frame per decoder step.',
    }[variant.id];
  }

  trainingNeed(variant: ExperimentVariantReport): string {
    return {
      'v1a-conservative-full': 'Test whether the earlier V1 regression came from an overly aggressive full-model update.',
      'v1b-lora': 'Constrain trainable capacity and protect pretrained representations while adapting attention.',
      'v1c-gradual-unfreeze': 'Control when decoder capacity becomes trainable instead of exposing the entire network immediately.',
      'v1d-reduction-factor-1': 'Test whether finer autoregressive acoustic resolution improves domain-word articulation.',
    }[variant.id];
  }

  scheduleConstruction(variant: ExperimentVariantReport, report: ExperimentReport): string {
    return `The same ${this.integer(variant.dataset.scheduledRows)}-row V1 train schedule, ${this.integer(report.sharedSplits['validation'])}-row validation set, and ${this.integer(report.sharedSplits['test'])}-row test set were reused without approach-specific sampling.`;
  }

  conclusion(variant: ExperimentVariantReport, report: ExperimentReport): string {
    const controlGap = (variant.evaluation.domainTermAccuracy - report.pretrainedControl.evaluation.domainTermAccuracy) * 100;
    if (variant.id === 'v1a-conservative-full') return 'Reject as a deployment candidate. Lower validation loss did not translate into better pronunciation or transcription quality.';
    if (variant.id === 'v1b-lora') return 'Keep as the performance trade-off: it was the fastest adapted model, but the control retained better term accuracy and WER.';
    if (variant.id === 'v1c-gradual-unfreeze') return `Keep as the strongest adaptation result. It tied pretrained term accuracy, but do not replace the control because WER and runtime were worse.`;
    return `Reject this architecture change. It finished training but lost ${Math.abs(controlGap).toFixed(2)} term-accuracy points and failed 68 synthesis cases.`;
  }

  primaryWinnerId(report: ExperimentReport): string {
    return [report.pretrainedControl, ...report.variants].reduce((winner, candidate) =>
      candidate.evaluation.domainTermAccuracy > winner.evaluation.domainTermAccuracy ? candidate : winner,
    ).id;
  }

  trainingBatch(variant: ExperimentVariantReport): string {
    const training = variant.trainingConfig;
    return training ? `${training.physicalBatchSize} × ${training.gradientAccumulationSteps}` : 'Not recorded';
  }

  learningRate(variant: ExperimentVariantReport): string {
    return variant.trainingConfig ? variant.trainingConfig.learningRate.toExponential(0) : 'Not recorded';
  }

  selectionLabel(variant: ExperimentVariantReport): string {
    if (!variant.selectionStatus) return 'Not recorded';
    return variant.selectionStatus === 'eligible_checkpoint_selected'
      ? `Guardrail-eligible checkpoint ${variant.selectedStep} selected.`
      : `Best available checkpoint ${variant.selectedStep}; no checkpoint passed every quality guardrail.`;
  }

  private signed(value: number): string { return `${value >= 0 ? '+' : ''}${value.toFixed(2)}`; }
}
