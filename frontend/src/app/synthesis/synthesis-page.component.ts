import { ChangeDetectionStrategy, Component } from '@angular/core';

import { AudioPlayerComponent } from '../audio-player/audio-player.component';
import { InferenceMetricsComponent } from '../metrics/inference-metrics.component';
import { ModelSelectorComponent } from '../model-selector/model-selector.component';
import { SynthesisFormComponent } from './synthesis-form.component';

@Component({
  selector: 'ovl-synthesis-page',
  standalone: true,
  imports: [
    AudioPlayerComponent,
    InferenceMetricsComponent,
    ModelSelectorComponent,
    SynthesisFormComponent,
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <section>
      <h2>Synthesis</h2>
      <p>The synthesis workflow will be implemented in a later revision.</p>
      <ovl-model-selector />
      <ovl-synthesis-form />
      <ovl-audio-player />
      <ovl-inference-metrics />
    </section>
  `,
})
export class SynthesisPageComponent {}

