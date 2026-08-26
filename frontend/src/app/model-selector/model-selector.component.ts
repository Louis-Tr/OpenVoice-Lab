import { ChangeDetectionStrategy, Component } from '@angular/core';

@Component({
  selector: 'ovl-model-selector',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <section aria-labelledby="model-selector-heading">
      <h3 id="model-selector-heading">Model and voice</h3>
      <p>Available voices and FP32 or quantized variants have not been loaded.</p>
    </section>
  `,
})
export class ModelSelectorComponent {}

