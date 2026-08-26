import { ChangeDetectionStrategy, Component } from '@angular/core';

import { BenchmarkResultsComponent } from './benchmark-results.component';

@Component({
  selector: 'ovl-benchmark-page',
  standalone: true,
  imports: [BenchmarkResultsComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <section>
      <h2>Benchmarks</h2>
      <p>Benchmark execution controls will be implemented later.</p>
      <ovl-benchmark-results />
    </section>
  `,
})
export class BenchmarkPageComponent {}

