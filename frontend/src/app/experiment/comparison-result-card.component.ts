import { ChangeDetectionStrategy, Component, Input } from '@angular/core';

import { ExperimentModelResult, ExperimentModelSummary } from './experiment.types';

@Component({
  selector: 'ovl-comparison-result-card',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  styleUrl: './comparison-result-card.component.css',
  template: `
    <article [class]="'result-card state-' + result.status" [attr.aria-label]="modelName()">
      <header>
        <div>
          <span>{{ modelRole() }}</span>
          <h4>{{ modelName() }}</h4>
        </div>
        <strong>{{ statusLabel() }}</strong>
      </header>

      @if (result.audioUrl) {
        <audio controls preload="metadata" [src]="result.audioUrl">
          Your browser does not support WAV playback.
        </audio>
      } @else if (result.status !== 'failure') {
        <div class="audio-placeholder" aria-hidden="true"><i></i><i></i><i></i><i></i></div>
      }

      @if (result.status === 'failure') {
        <p class="error" role="alert">{{ result.error }}</p>
      }

      @if (result.transcript) {
        <div class="transcript">
          <span>ASR transcription</span>
          <p>{{ result.transcript }}</p>
        </div>
      }

      @if (result.targetTerms; as terms) {
        <div class="term-row">
          @for (term of terms.correct; track term) {<span class="correct">{{ term }} · correct</span>}
          @for (term of terms.incorrect; track term) {<span class="missed">{{ term }} · missed</span>}
        </div>
      }

      @if (result.metrics; as metrics) {
        <dl>
          <div><dt>Term accuracy</dt><dd>{{ percent(result.targetTerms?.accuracy) }}</dd></div>
          <div><dt>WER</dt><dd>{{ percent(result.wordErrorRate) }}</dd></div>
          <div><dt>Inference</dt><dd>{{ number(metrics.inferenceMs, 0) }} ms</dd></div>
          <div><dt>RTF</dt><dd>{{ number(metrics.realTimeFactor, 4) }}</dd></div>
          <div><dt>Process RSS</dt><dd>{{ number(metrics.processMemoryMb, 0) }} MB</dd></div>
          <div><dt>State</dt><dd>{{ metrics.warm ? 'Warm' : 'Cold' }} CPU</dd></div>
        </dl>
      }
    </article>
  `,
})
export class ComparisonResultCardComponent {
  @Input({ required: true }) result!: ExperimentModelResult;
  @Input({ required: true }) models: readonly ExperimentModelSummary[] = [];

  modelName(): string {
    return this.models.find((model) => model.id === this.result.modelId)?.name ?? this.result.modelId;
  }

  modelRole(): string {
    return this.models.find((model) => model.id === this.result.modelId)?.role ?? 'model';
  }

  statusLabel(): string {
    const labels: Record<ExperimentModelResult['status'], string> = {
      queued: 'Queued',
      loading: 'Loading',
      synthesizing: 'Generating',
      audio_ready: 'Audio ready',
      transcribing: 'Scoring',
      success: 'Complete',
      failure: 'Failed',
    };
    return labels[this.result.status];
  }

  number(value: number, digits: number): string {
    return value.toFixed(digits);
  }

  percent(value: number | null | undefined): string {
    return value === null || value === undefined ? '—' : `${(value * 100).toFixed(1)}%`;
  }
}
