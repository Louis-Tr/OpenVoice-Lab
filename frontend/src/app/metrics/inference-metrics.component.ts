import { ChangeDetectionStrategy, Component } from '@angular/core';

@Component({
  selector: 'ovl-inference-metrics',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <section aria-labelledby="inference-metrics-heading">
      <h3 id="inference-metrics-heading">Inference metrics</h3>
      <p>Latency, RTF, memory, and model metadata are not available yet.</p>
    </section>
  `,
})
export class InferenceMetricsComponent {}

