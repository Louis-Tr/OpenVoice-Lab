import { ChangeDetectionStrategy, Component, Input } from '@angular/core';

import { BenchmarkAggregate } from '../api/benchmark-api.service';

export type BenchmarkResultsState = 'empty' | 'loading' | 'ready' | 'error';

@Component({
  selector: 'ovl-benchmark-results',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  styleUrl: './benchmark-results.component.css',
  template: `
    <section class="comparison-card" aria-labelledby="benchmark-results-heading">
      <div class="card-heading">
        <span>02 / COMPARISON</span>
        @if (state === 'ready') {
          <span class="ready-state"><i aria-hidden="true"></i> Complete</span>
        }
      </div>
      <h3 id="benchmark-results-heading">Model comparison</h3>

      @if (state === 'loading') {
        <div class="empty-state" aria-live="polite">
          <strong>Evaluation in progress.</strong>
          <span>Results appear here when every model has processed the fixed corpus.</span>
        </div>
      } @else if (state === 'error') {
        <div class="empty-state error-state">
          <strong>No comparison was produced.</strong>
          <span>The benchmark failed before aggregate results were available.</span>
        </div>
      } @else if (aggregates.length > 0) {
        <div class="table-scroll" tabindex="0" aria-label="Scrollable model comparison table">
          <table>
            <caption>Deployment metrics by model configuration</caption>
            <thead>
              <tr>
                <th scope="col">Model</th>
                <th scope="col">Avg latency</th>
                <th scope="col">P95</th>
                <th scope="col">Avg RTF</th>
                <th scope="col">Peak RSS</th>
                <th scope="col">Failures</th>
              </tr>
            </thead>
            <tbody>
              @for (aggregate of aggregates; track aggregate.modelId) {
                <tr>
                  <th scope="row">
                    <strong>{{ aggregate.precision }}</strong>
                    <span>{{ aggregate.modelId }}</span>
                  </th>
                  <td>{{ formatMilliseconds(aggregate.averageLatencyMs) }}</td>
                  <td>{{ formatMilliseconds(aggregate.p95LatencyMs) }}</td>
                  <td>{{ formatRatio(aggregate.averageRealTimeFactor) }}</td>
                  <td>{{ formatMemory(aggregate.peakMemoryMb) }}</td>
                  <td [class.has-failures]="aggregate.failureCount > 0">
                    {{ aggregate.failureCount }}
                  </td>
                </tr>
              }
            </tbody>
          </table>
        </div>
        <p class="decision-note">
          Lower latency and RTF favor responsiveness. Peak RSS shows the memory cost of each isolated model process.
        </p>
      } @else {
        <div class="empty-state">
          <strong>No benchmark yet.</strong>
          <span>Run the fixed evaluation corpus to compare deployed model variants.</span>
        </div>
      }
    </section>
  `,
})
export class BenchmarkResultsComponent {
  @Input() aggregates: readonly BenchmarkAggregate[] = [];
  @Input() state: BenchmarkResultsState = 'empty';

  formatMilliseconds(value: number | null): string {
    return value === null ? '—' : `${this.formatNumber(value, 3)} ms`;
  }

  formatRatio(value: number | null): string {
    return value === null ? '—' : this.formatNumber(value, 6);
  }

  formatMemory(value: number | null): string {
    return value === null ? '—' : `${this.formatNumber(value, 3)} MB`;
  }

  private formatNumber(value: number, digits: number): string {
    return new Intl.NumberFormat('en-US', {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    }).format(value);
  }
}
