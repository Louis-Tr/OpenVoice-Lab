import { ChangeDetectionStrategy, Component, Input } from '@angular/core';

import { InferenceMetrics } from '../synthesis/synthesis.types';

@Component({
  selector: 'ovl-inference-metrics',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  styleUrl: './inference-metrics.component.css',
  template: `
    <section class="metrics-card" aria-labelledby="inference-metrics-heading" aria-live="polite">
      <div class="card-heading">
        <span>03 / MEASURE</span>
        @if (metrics) {
          <span class="runtime-state">
            <i aria-hidden="true"></i>
            {{ metrics.warm ? 'Warm' : 'Cold' }}
          </span>
        }
      </div>
      <h3 id="inference-metrics-heading">Inference metrics</h3>

      @if (metrics; as measured) {
        <dl>
          <div>
            <dt>Inference</dt>
            <dd>{{ formatMilliseconds(measured.inferenceMs) }}<span>ms</span></dd>
          </div>
          <div>
            <dt>Audio duration</dt>
            <dd>{{ formatMilliseconds(measured.audioDurationMs) }}<span>ms</span></dd>
          </div>
          <div>
            <dt>Real-time factor</dt>
            <dd>{{ formatRtf(measured.realTimeFactor) }}<span>RTF</span></dd>
          </div>
          <div>
            <dt>Process memory</dt>
            <dd>{{ formatMemory(measured.memoryMb) }}<span>MB</span></dd>
          </div>
          <div>
            <dt>Model load</dt>
            <dd>{{ formatMilliseconds(measured.modelLoadMs) }}<span>ms</span></dd>
          </div>
          <div>
            <dt>Variant</dt>
            <dd class="variant">{{ measured.modelVariant.toUpperCase() }}</dd>
          </div>
        </dl>
        <p class="definition">
          RTF = inference time ÷ audio duration. Lower than 1.000 means faster than real time.
        </p>
      } @else {
        <div class="empty-metrics">
          <p>No measurements yet.</p>
          <span>Generate audio to measure this model execution.</span>
        </div>
      }
    </section>
  `,
})
export class InferenceMetricsComponent {
  @Input() metrics: InferenceMetrics | null = null;

  formatMilliseconds(value: number): string {
    return Math.round(value).toLocaleString('en-US');
  }

  formatMemory(value: number): string {
    return Math.round(value).toLocaleString('en-US');
  }

  formatRtf(value: number): string {
    return value.toFixed(3);
  }
}
