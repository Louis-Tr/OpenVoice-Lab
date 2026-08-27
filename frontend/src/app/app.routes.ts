import { Routes } from '@angular/router';

import { BenchmarkPageComponent } from './benchmark/benchmark-page.component';
import { SynthesisPageComponent } from './synthesis/synthesis-page.component';

export const routes: Routes = [
  { path: '', component: SynthesisPageComponent, title: 'Synthesis | OpenVoice Lab' },
  {
    path: 'benchmarks',
    component: BenchmarkPageComponent,
    title: 'Benchmarks | OpenVoice Lab',
  },
  {
    path: 'experiments/stage11',
    loadComponent: () =>
      import('./experiment/stage11-experiment-page.component').then(
        (module) => module.Stage11ExperimentPageComponent,
      ),
    title: 'Stage 11 Experiment | OpenVoice Lab',
  },
  { path: '**', redirectTo: '' },
];
