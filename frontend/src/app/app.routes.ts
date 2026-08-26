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
  { path: '**', redirectTo: '' },
];

