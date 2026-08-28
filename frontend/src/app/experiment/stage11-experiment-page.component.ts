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
        <p class="eyebrow"><i aria-hidden="true"></i> Stage 11 / controlled SpeechT5 adaptation</p>
        <div class="hero-layout">
          <div>
            <h2 id="experiment-title">Three fine-tunes.<br /><span>One honest result.</span></h2>
            <p class="intro">See the training decisions, inspect the measured tradeoffs, then run the pretrained control and adapted models on the same sentence.</p>
            <div class="hero-actions">
              <a class="primary-link" href="#live-lab">Run the live comparison</a>
              <a class="text-link" href="#experiment-design">Inspect the experiment <span aria-hidden="true">→</span></a>
            </div>
          </div>
          <ol class="hiring-tour" aria-label="Suggested experiment walkthrough">
            <li><span>01</span><div><strong>Read the outcome</strong><small>Expectation versus actual evidence</small></div></li>
            <li><span>02</span><div><strong>Challenge it live</strong><small>Fixture or your own medical terms</small></div></li>
            <li><span>03</span><div><strong>Audit the method</strong><small>Data, training, checkpoints, hashes</small></div></li>
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
          <div><strong>Stage 11 evidence is unavailable.</strong><p>{{ error() }}</p></div>
          <button type="button" (click)="load()">Retry</button>
        </section>
      } @else if (report(); as evidence) {
        <section class="integrity" aria-label="Experiment integrity">
          <div><span>Dataset lock</span><strong>{{ evidence.integrity.datasetLockVerified ? 'Verified' : 'Failed' }}</strong></div>
          <div><span>Checkpoints</span><strong>{{ evidence.integrity.checkpointCount }} verified</strong></div>
          <div><span>Final models</span><strong>{{ evidence.integrity.finalArtifactHashesVerified ? 'Verified' : 'Failed' }}</strong></div>
          <div><span>RunPod state</span><strong>{{ evidence.integrity.allPodsTerminated ? 'All terminated' : 'Active' }}</strong></div>
        </section>

        <section class="outcome surface" aria-labelledby="measured-results-title">
          <div class="section-label">Expectation → measurement → decision</div>
          <div class="section-heading">
            <h3 id="measured-results-title">The control won.<br />Adaptation regressed.</h3>
            <p>{{ evidence.headline }} The experiment keeps every negative result because it changes the engineering decision.</p>
          </div>

          <div class="hypothesis-grid">
            @for (variant of evidence.variants; track variant.id) {
              <article [class.best-outcome]="variant.id === 'v3-replay'">
                <div class="card-topline"><span>{{ variant.name }}</span><strong>{{ outcomeLabel(variant) }}</strong></div>
                <p><b>Expected</b>{{ expectation(variant) }}</p>
                <p><b>Measured</b>{{ measuredOutcome(variant, evidence) }}</p>
              </article>
            }
          </div>

          <div class="table-scroll" tabindex="0" aria-label="Historical Stage 11 evaluation results including the pretrained control">
            <table>
              <caption>{{ evidence.runtimeLabel }} · common {{ evidence.pretrainedControl.evaluation.caseCount }}-case test set</caption>
              <thead>
                <tr><th scope="col">Model</th><th scope="col">Training</th><th scope="col">Term accuracy</th><th scope="col">WER</th><th scope="col">Avg inference</th><th scope="col">RTF</th><th scope="col">Peak GPU</th><th scope="col">Failures</th></tr>
              </thead>
              <tbody>
                <tr class="control-row" [class.winner]="primaryWinnerId(evidence) === evidence.pretrainedControl.id">
                  <th scope="row"><strong>{{ evidence.pretrainedControl.name }}</strong><span>unadapted control · {{ evidence.pretrainedControl.hardware }} · pinned {{ shortHash(evidence.pretrainedControl.revision) }}</span></th>
                  <td>{{ evidence.pretrainedControl.trainingSteps }} steps</td>
                  <td>{{ percent(evidence.pretrainedControl.evaluation.domainTermAccuracy) }}</td>
                  <td>{{ percent(evidence.pretrainedControl.evaluation.wordErrorRate) }}</td>
                  <td>{{ integer(evidence.pretrainedControl.evaluation.averageInferenceMs) }} ms</td>
                  <td>{{ fixed(evidence.pretrainedControl.evaluation.averageRealTimeFactor, 4) }}</td>
                  <td>{{ integer(evidence.pretrainedControl.evaluation.peakGpuMemoryMb) }} MB</td>
                  <td>{{ evidence.pretrainedControl.evaluation.failureCount }}</td>
                </tr>
                @for (variant of evidence.variants; track variant.id) {
                  <tr [class.winner]="primaryWinnerId(evidence) === variant.id">
                    <th scope="row"><strong>{{ variant.name }}</strong><span>{{ variant.dataset.strategy }} · best step {{ variant.bestStep }}</span></th>
                    <td>{{ fixed(variant.trainingSeconds / 60, 1) }} min</td>
                    <td>{{ percent(variant.evaluation.domainTermAccuracy) }}</td>
                    <td>{{ percent(variant.evaluation.wordErrorRate) }}</td>
                    <td>{{ integer(variant.evaluation.averageInferenceMs) }} ms</td>
                    <td>{{ fixed(variant.evaluation.averageRealTimeFactor, 4) }}</td>
                    <td>{{ integer(variant.evaluation.peakGpuMemoryMb) }} MB</td>
                    <td>{{ variant.evaluation.failureCount }}</td>
                  </tr>
                }
              </tbody>
            </table>
          </div>
          <p class="evidence-note"><strong>Measurement boundary:</strong> all four rows use the same locked {{ evidence.pretrainedControl.evaluation.caseCount }}-case test manifest, pinned speaker source, vocoder, ASR revision, and secure RTX 4090 evaluation path. The pretrained control was evaluated without training; live CPU measurements below remain separate.</p>
        </section>

        <ovl-live-comparison [models]="models()" [fixtures]="fixtures()" />

        <section id="experiment-design" class="method surface" aria-labelledby="pipeline-title">
          <div class="section-label">Data engineering</div>
          <div class="section-heading compact-heading">
            <h3 id="pipeline-title">Same source. Purpose-built schedules.</h3>
            <p>The source corpus was cleaned once. Only the exposure schedule changed, isolating the dataset strategy as the experiment variable.</p>
          </div>

          <div class="audit-strip" aria-label="Dataset preparation audit">
            <div><span>Approved audio</span><strong>{{ integer(evidence.dataAudit.uniqueAudioFiles) }}</strong><small>16 kHz mono files</small></div>
            <div><span>Audio failures</span><strong>{{ evidence.dataAudit.audioVerificationFailureCount }}</strong><small>after verification</small></div>
            <div><span>Leakage intersections</span><strong>{{ evidence.dataAudit.leakageIntersectionCount }}</strong><small>across {{ evidence.dataAudit.leakageIdentityFields.length }} identity checks</small></div>
            <div><span>Shared test</span><strong>{{ integer(evidence.sharedSplits['test']) }}</strong><small>identical cases per variant</small></div>
          </div>

          <div class="pipeline" aria-label="Stage 11 data and training pipeline">
            <span>Immutable intake</span><i aria-hidden="true">→</i><span>Audio + text cleanup</span><i aria-hidden="true">→</i><span>Quality + dedupe</span><i aria-hidden="true">→</i><span>Leakage-safe splits</span><i aria-hidden="true">→</i><span>Variant schedules</span><i aria-hidden="true">→</i><span>Locked manifests</span>
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
                  <div class="summary-metrics"><span>{{ percent(termExposureRate(variant)) }} term rows</span><span>{{ integer(variant.dataset.repeatedExposures) }} repeats</span><b>{{ outcomeLabel(variant) }}</b></div>
                </summary>
                <div class="variant-detail">
                  <div class="story-column">
                    <p><b>Training need</b>{{ trainingNeed(variant) }}</p>
                    <p><b>Schedule construction</b>{{ scheduleConstruction(variant, evidence) }}</p>
                    <p><b>Engineering conclusion</b>{{ conclusion(variant, evidence) }}</p>
                  </div>
                  <dl class="variant-stats">
                    <div><dt>Scheduled rows</dt><dd>{{ integer(variant.dataset.scheduledRows) }}</dd></div>
                    <div><dt>Unique rows</dt><dd>{{ integer(variant.dataset.uniqueSourceRows) }}</dd></div>
                    <div><dt>Term rows</dt><dd>{{ integer(variant.dataset.rowsWithTerms) }}</dd></div>
                    <div><dt>Speakers</dt><dd>{{ variant.dataset.uniqueSpeakers }}</dd></div>
                    <div><dt>Audio exposure</dt><dd>{{ fixed(variant.dataset.durationHours, 2) }} h</dd></div>
                    <div><dt>Max speaker share</dt><dd>{{ percent(variant.dataset.maximumSpeakerShare) }}</dd></div>
                    <div><dt>Best eval loss</dt><dd>{{ fixed(variant.bestValidationLoss, 6) }}</dd></div>
                    <div><dt>Best / final step</dt><dd>{{ variant.bestStep }} / {{ variant.finalStep }}</dd></div>
                    <div><dt>Checkpoints</dt><dd>{{ variant.checkpointSteps.length }} × verified</dd></div>
                    <div><dt>Estimated cost</dt><dd>USD {{ fixed(variant.estimatedCostUsd, 2) }}</dd></div>
                  </dl>
                  <p class="pool-line"><strong>Exposure pools</strong> {{ poolLabel(variant) }}</p>
                  <p class="pool-line"><strong>Tracked occurrences</strong> {{ categoryLabel(variant) }}</p>
                  <p class="pool-line"><strong>Train manifest</strong> {{ shortHash(variant.dataset.manifestSha256['train']) }}</p>
                </div>
              </details>
            }
          </div>
        </section>

        <section class="training surface" aria-labelledby="training-title">
          <div class="section-label">Training execution</div>
          <div class="section-heading compact-heading">
            <h3 id="training-title">Controlled where it matters.</h3>
            <p>One secure RTX 4090 per variant, concurrent execution, identical model revisions and optimization settings.</p>
          </div>
          <dl class="config-grid">
            <div><dt>Precision</dt><dd>{{ evidence.training.precision }}</dd></div>
            <div><dt>Physical batch</dt><dd>{{ evidence.training.physicalBatchSize }}</dd></div>
            <div><dt>Accumulation</dt><dd>{{ evidence.training.gradientAccumulationSteps }}</dd></div>
            <div><dt>Effective batch</dt><dd>{{ evidence.training.effectiveBatchSize }}</dd></div>
            <div><dt>Maximum steps</dt><dd>{{ integer(evidence.training.maximumSteps) }}</dd></div>
            <div><dt>Learning rate</dt><dd>{{ evidence.training.learningRate }}</dd></div>
            <div><dt>Warmup</dt><dd>{{ evidence.training.warmupSteps }} steps</dd></div>
            <div><dt>Validation</dt><dd>Every {{ evidence.training.evaluationSteps }}</dd></div>
          </dl>
          <details class="method-disclosure training-disclosure">
            <summary><span>Open optimization and recovery policy</span><small>Stopping, clipping, checkpoints, source revisions</small></summary>
            <div class="disclosure-body two-column-copy">
              <div><strong>Optimization</strong><p>Gradient checkpointing enabled, maximum gradient norm {{ evidence.training.maximumGradientNorm }}, seed {{ evidence.training.seed }}, and a frozen {{ evidence.training.nominalEpochs }}-epoch / {{ evidence.training.maximumSteps }}-step ceiling.</p></div>
              <div><strong>Early stopping</strong><p>Validation loss monitored every {{ evidence.training.evaluationSteps }} steps with patience {{ evidence.training.earlyStoppingPatience }} and minimum improvement {{ evidence.training.earlyStoppingThreshold }}.</p></div>
              <div><strong>Recovery</strong><p>Recoverable checkpoints every 125 optimizer steps. All 24 checkpoints include state, provenance, hashes, and completion markers.</p></div>
              <div><strong>Model provenance</strong><p>SpeechT5 {{ shortHash(evidence.sourceModelRevisions['tts']) }}, HiFi-GAN {{ shortHash(evidence.sourceModelRevisions['vocoder']) }}, Whisper {{ shortHash(evidence.sourceModelRevisions['asr']) }}.</p></div>
            </div>
          </details>
        </section>

        <ovl-training-loss-chart [variants]="evidence.variants" />

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
                <div><dt>Pretrained control pod</dt><dd>{{ evidence.pretrainedControl.podId }} · {{ evidence.pretrainedControl.evaluation.caseCount }} cases · artifact {{ shortHash(evidence.pretrainedControl.artifactManifestSha256) }}</dd></div>
                @for (variant of evidence.variants; track variant.id) {
                  <div><dt>{{ variant.name }} pod</dt><dd>{{ variant.podId }} · best step {{ variant.bestStep }} · {{ variant.stoppedEarly ? 'early stopped' : 'completed max steps' }}</dd></div>
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
  percent(value: number): string { return `${(value * 100).toFixed(2)}%`; }
  integer(value: number): string { return new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(value); }
  shortHash(value: string): string { return `${value.slice(0, 12)}…`; }

  outcomeLabel(variant: ExperimentVariantReport): string {
    return { 'v1-baseline': 'Adapted reference', 'v2-term-balance': 'Hypothesis rejected', 'v3-replay': 'Best adapted' }[variant.id];
  }

  expectation(variant: ExperimentVariantReport): string {
    return {
      'v1-baseline': 'Establish a stable adaptation reference without deliberately changing term exposure.',
      'v2-term-balance': 'Increasing medical-term exposure should improve domain-term accuracy.',
      'v3-replay': 'A 4:4 term/general replay block should improve terms while retaining broader speech behavior.',
    }[variant.id];
  }

  measuredOutcome(variant: ExperimentVariantReport, report: ExperimentReport): string {
    const baseline = report.variants[0];
    const controlGap = (variant.evaluation.domainTermAccuracy - report.pretrainedControl.evaluation.domainTermAccuracy) * 100;
    if (variant.id === 'v1-baseline') return `${this.percent(variant.evaluation.domainTermAccuracy)} term accuracy, ${this.signed(controlGap)} points versus pretrained.`;
    const points = (variant.evaluation.domainTermAccuracy - baseline.evaluation.domainTermAccuracy) * 100;
    if (variant.id === 'v2-term-balance') return `${this.signed(points)} points versus V1 and ${this.signed(controlGap)} versus pretrained despite ${this.integer(variant.dataset.repeatedExposures)} repeats.`;
    const rtf = ((variant.evaluation.averageRealTimeFactor / baseline.evaluation.averageRealTimeFactor) - 1) * 100;
    return `${this.signed(points)} points versus V1, ${this.signed(controlGap)} versus pretrained, with RTF ${this.signed(rtf)}%.`;
  }

  strategyTitle(variant: ExperimentVariantReport): string {
    return { 'v1-baseline': 'Uniform baseline', 'v2-term-balance': 'Term-balanced exposure', 'v3-replay': 'Controlled replay' }[variant.id];
  }

  strategyDescription(variant: ExperimentVariantReport): string {
    return {
      'v1-baseline': 'Preserves the approved source distribution.',
      'v2-term-balance': 'Reweights every scheduled exposure toward tracked medical terms.',
      'v3-replay': 'Locks each eight-row block to four term-balanced and four general replay rows.',
    }[variant.id];
  }

  trainingNeed(variant: ExperimentVariantReport): string {
    return {
      'v1-baseline': 'Measure what ordinary supervised adaptation changes before adding sampling intervention.',
      'v2-term-balance': 'Stress whether concentrated domain vocabulary exposure alone fixes pronunciation.',
      'v3-replay': 'Preserve general examples while guaranteeing domain-term presence in every optimizer block.',
    }[variant.id];
  }

  scheduleConstruction(variant: ExperimentVariantReport, report: ExperimentReport): string {
    if (variant.id === 'v1-baseline') return `${this.integer(variant.dataset.uniqueSourceRows)} approved rows were scheduled almost once each.`;
    if (variant.id === 'v2-term-balance') return `${this.integer(variant.dataset.uniqueSourceRows)} unique term-bearing rows were deterministically repeated to fill ${this.integer(variant.dataset.scheduledRows)} exposures.`;
    return `${this.integer(variant.dataset.scheduledRows / report.dataAudit.scheduleBlockSize)} locked blocks combined equal term-balanced and replay pools without row-level shuffle.`;
  }

  conclusion(variant: ExperimentVariantReport, report: ExperimentReport): string {
    if (variant.id === 'v1-baseline') return 'Keep as the adapted reference, not a deployment candidate: it reached the lowest validation loss but regressed sharply against pretrained.';
    if (variant.id === 'v2-term-balance') return 'Do not deploy this sampling policy. More target exposure reduced both term accuracy and overall WER performance.';
    const baseline = report.variants[0];
    const points = (variant.evaluation.domainTermAccuracy - baseline.evaluation.domainTermAccuracy) * 100;
    const controlGap = (variant.evaluation.domainTermAccuracy - report.pretrainedControl.evaluation.domainTermAccuracy) * 100;
    return `Best adapted result at ${this.signed(points)} points over V1, but do not prefer it over pretrained: the control remains ${Math.abs(controlGap).toFixed(2)} points ahead.`;
  }

  primaryWinnerId(report: ExperimentReport): string {
    return [report.pretrainedControl, ...report.variants].reduce((winner, candidate) =>
      candidate.evaluation.domainTermAccuracy > winner.evaluation.domainTermAccuracy ? candidate : winner,
    ).id;
  }

  termExposureRate(variant: ExperimentVariantReport): number { return variant.dataset.rowsWithTerms / variant.dataset.scheduledRows; }

  poolLabel(variant: ExperimentVariantReport): string {
    return Object.entries(variant.dataset.sourcePoolCounts).map(([name, count]) => `${name}: ${this.integer(count)}`).join(' · ');
  }

  categoryLabel(variant: ExperimentVariantReport): string {
    return Object.entries(variant.dataset.termCategoryOccurrences).map(([name, count]) => `${name}: ${this.integer(count)}`).join(' · ');
  }

  private signed(value: number): string { return `${value >= 0 ? '+' : ''}${value.toFixed(2)}`; }
}
