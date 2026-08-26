import { ChangeDetectionStrategy, Component } from '@angular/core';

@Component({
  selector: 'ovl-synthesis-form',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <section aria-labelledby="synthesis-form-heading">
      <h3 id="synthesis-form-heading">Synthesis request</h3>
      <p>Text entry and request submission controls are not implemented yet.</p>
    </section>
  `,
})
export class SynthesisFormComponent {}

