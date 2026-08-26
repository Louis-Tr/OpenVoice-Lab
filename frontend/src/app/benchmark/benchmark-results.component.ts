import { ChangeDetectionStrategy, Component } from '@angular/core';

@Component({
  selector: 'ovl-benchmark-results',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <section aria-labelledby="benchmark-results-heading">
      <h3 id="benchmark-results-heading">Aggregate results</h3>
      <p>No benchmark has been run.</p>
    </section>
  `,
})
export class BenchmarkResultsComponent {}

