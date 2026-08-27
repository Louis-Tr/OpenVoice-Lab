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
    <section class="experiment-page" aria-labelledby="experiment-title">
      <header class="hero">
        <p class="eyebrow"><i aria-hidden="true"></i> Stage 11 / SpeechT5 adaptation</p>
        <h2 id="experiment-title">Training evidence.<br /><span>Testable in public.</span></h2>
        <p class="intro">A locked three-variant fine-tuning experiment, presented with its failures, provenance, and real CPU inference.</p>
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
          <div><span>RunPod state</span><strong>{{ evidence.integrity.allPodsTerminated ? 'Terminated' : 'Active' }}</strong></div>
        </section>

        <section class="conclusion" aria-labelledby="measured-results-title">
          <div class="section-label">Measured conclusion</div>
          <div class="conclusion-heading">
            <h3 id="measured-results-title">V3 won pronunciation.<br />Not runtime.</h3>
            <p>{{ evidence.headline }}</p>
          </div>
          <div class="table-scroll" tabindex="0" aria-label="Historical Stage 11 evaluation results">
            <table>
              <caption>{{ evidence.runtimeLabel }}</caption>
              <thead><tr><th scope="col">Variant</th><th scope="col">Best step</th><th scope="col">Eval loss</th><th scope="col">Term accuracy</th><th scope="col">WER</th><th scope="col">RTF</th></tr></thead>
              <tbody>
                @for (variant of evidence.variants; track variant.id) {
                  <tr [class.winner]="variant.id === 'v3-replay'">
                    <th scope="row"><strong>{{ variant.name }}</strong><span>{{ variant.dataset.strategy }}</span></th>
                    <td>{{ variant.bestStep }}</td>
                    <td>{{ fixed(variant.bestValidationLoss, 6) }}</td>
                    <td>{{ percent(variant.evaluation.domainTermAccuracy) }}</td>
                    <td>{{ percent(variant.evaluation.wordErrorRate) }}</td>
                    <td>{{ fixed(variant.evaluation.averageRealTimeFactor, 4) }}</td>
                  </tr>
                }
              </tbody>
            </table>
          </div>
          <p class="evidence-note">These are historical RTX 4090 measurements from the shared locked 662-case test set. Live CPU measurements below are separate.</p>
        </section>

        <ovl-live-comparison [models]="models()" [fixtures]="fixtures()" />

        <section class="pipeline-section" aria-labelledby="pipeline-title">
          <div class="section-label">Training pipeline</div>
          <h3 id="pipeline-title">One workload. Three controlled schedules.</h3>
          <div class="pipeline" aria-label="Stage 11 training pipeline">
            <span>Raw medical speech</span><i aria-hidden="true">↓</i><span>Deterministic processing</span><i aria-hidden="true">↓</i><span>Locked datasets</span><i aria-hidden="true">↓</i><span>3 × secure RTX 4090</span><i aria-hidden="true">↓</i><span>125-step checkpoints</span><i aria-hidden="true">↓</i><span>Shared ASR evaluation</span>
          </div>
          <div class="dataset-grid">
            @for (variant of evidence.variants; track variant.id) {
              <article>
                <span>{{ variant.id }}</span>
                <h4>{{ strategyTitle(variant) }}</h4>
                <p>{{ strategyDescription(variant) }}</p>
                <dl>
                  <div><dt>Exposures</dt><dd>{{ integer(variant.dataset.scheduledRows) }}</dd></div>
                  <div><dt>Unique rows</dt><dd>{{ integer(variant.dataset.uniqueSourceRows) }}</dd></div>
                  <div><dt>Repeats</dt><dd>{{ integer(variant.dataset.repeatedExposures) }}</dd></div>
                  <div><dt>Audio</dt><dd>{{ fixed(variant.dataset.durationHours, 2) }} h</dd></div>
                </dl>
                <small>{{ poolLabel(variant) }}</small>
              </article>
            }
          </div>
        </section>

        <section class="training-section" aria-labelledby="training-title">
          <div class="section-label">Frozen configuration</div>
          <h3 id="training-title">No tuning after the result.</h3>
          <dl class="config-grid">
            <div><dt>Precision</dt><dd>{{ evidence.training.precision }}</dd></div>
            <div><dt>Physical batch</dt><dd>{{ evidence.training.physicalBatchSize }}</dd></div>
            <div><dt>Accumulation</dt><dd>{{ evidence.training.gradientAccumulationSteps }}</dd></div>
            <div><dt>Effective batch</dt><dd>{{ evidence.training.effectiveBatchSize }}</dd></div>
            <div><dt>Max steps</dt><dd>{{ integer(evidence.training.maximumSteps) }}</dd></div>
            <div><dt>Learning rate</dt><dd>{{ evidence.training.learningRate }}</dd></div>
            <div><dt>Warmup</dt><dd>{{ evidence.training.warmupSteps }}</dd></div>
            <div><dt>Validation</dt><dd>Every {{ evidence.training.evaluationSteps }}</dd></div>
            <div><dt>Clip norm</dt><dd>{{ evidence.training.maximumGradientNorm }}</dd></div>
            <div><dt>Seed</dt><dd>{{ evidence.training.seed }}</dd></div>
          </dl>
        </section>

        <ovl-training-loss-chart [variants]="evidence.variants" />

        <section class="provenance" aria-labelledby="provenance-title">
          <div class="section-label">Reproducibility</div>
          <h3 id="provenance-title">The evidence stays attached.</h3>
          <div class="proof-grid">
            <div><span>Total GPU time</span><strong>{{ fixed(evidence.totalGpuHours, 4) }} hours</strong></div>
            <div><span>Estimated cost</span><strong>USD {{ fixed(evidence.estimatedTotalCostUsd, 2) }}</strong></div>
            <div><span>Training resumptions</span><strong>{{ evidence.trainingResumptions.length }}</strong></div>
            <div><span>Configuration</span><strong>{{ shortHash(evidence.configurationSha256) }}</strong></div>
          </div>
          <details>
            <summary>Open run provenance</summary>
            <dl class="hash-list">
              <div><dt>Run ID</dt><dd>{{ evidence.runId }}</dd></div>
              <div><dt>Dataset lock</dt><dd>{{ evidence.datasetLockSha256 }}</dd></div>
              @for (variant of evidence.variants; track variant.id) {
                <div><dt>{{ variant.name }} pod</dt><dd>{{ variant.podId }} · best step {{ variant.bestStep }} · {{ variant.stoppedEarly ? 'early stopped' : 'completed max steps' }}</dd></div>
              }
            </dl>
          </details>
          @if (evidence.incidents.length > 0) {
            <div class="incident-list">
              @for (incident of evidence.incidents; track incident.id) {
                <article><strong>{{ incident.id }}</strong><p>{{ incident.impact }} {{ incident.resolution }}</p><span>Training restart: {{ incident.trainingRestart ? 'yes' : 'no' }}</span></article>
              }
            </div>
          }
        </section>
      }
    </section>
  `,
})
export class Stage11ExperimentPageComponent implements OnInit {
  readonly state = signal<'loading' | 'ready' | 'error'>('loading');
  readonly error = signal('');
  readonly report = signal<ExperimentReport | null>(null);
  readonly models = signal<readonly ExperimentModelSummary[]>([]);
  readonly fixtures = signal<readonly ExperimentFixture[]>([]);

  constructor(private readonly api: ExperimentApiService) {}

  ngOnInit(): void {
    this.load();
  }

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
  integer(value: number): string { return new Intl.NumberFormat('en-US').format(value); }
  shortHash(value: string): string { return `${value.slice(0, 12)}…`; }

  strategyTitle(variant: ExperimentVariantReport): string {
    return { 'v1-baseline': 'Uniform baseline', 'v2-term-balance': 'Term-balanced exposure', 'v3-replay': 'Controlled replay' }[variant.id];
  }

  strategyDescription(variant: ExperimentVariantReport): string {
    return {
      'v1-baseline': 'Preserves the source distribution with one repeated exposure.',
      'v2-term-balance': 'Reweights every exposure toward rows containing tracked medical terms.',
      'v3-replay': 'Locks every eight-row block to four term-balanced and four replay rows.',
    }[variant.id];
  }

  poolLabel(variant: ExperimentVariantReport): string {
    return Object.entries(variant.dataset.sourcePoolCounts).map(([name, count]) => `${name}: ${this.integer(count)}`).join(' · ');
  }
}
