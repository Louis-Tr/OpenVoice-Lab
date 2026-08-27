import { ChangeDetectionStrategy, Component, Input } from '@angular/core';

import { ExperimentVariantReport } from './experiment.types';

@Component({
  selector: 'ovl-training-loss-chart',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  styleUrl: './training-loss-chart.component.css',
  template: `
    <section class="chart-card" aria-labelledby="validation-loss-title">
      <div class="section-label">Validation history</div>
      <div class="title-row">
        <div>
          <h3 id="validation-loss-title">Loss by optimizer step</h3>
          <p>Eight shared evaluation points. Lower is better.</p>
        </div>
        <ul class="legend" aria-label="Chart legend">
          @for (variant of variants; track variant.id; let index = $index) {
            <li><i [class]="'series-' + index" aria-hidden="true"></i>{{ variant.name }}</li>
          }
        </ul>
      </div>

      <svg
        viewBox="0 0 720 260"
        role="img"
        aria-labelledby="validation-loss-title validation-loss-description"
      >
        <desc id="validation-loss-description">
          Line chart showing validation loss for all three SpeechT5 variants at steps 125 through 1000.
        </desc>
        @for (tick of yTicks(); track tick) {
          <line x1="58" x2="700" [attr.y1]="y(tick)" [attr.y2]="y(tick)" class="grid" />
          <text x="50" [attr.y]="y(tick) + 4" text-anchor="end">{{ fixed(tick, 2) }}</text>
        }
        @for (step of stepTicks(); track step) {
          <text [attr.x]="x(step)" y="245" text-anchor="middle">{{ step }}</text>
        }
        @for (variant of variants; track variant.id; let index = $index) {
          <polyline [attr.points]="points(variant)" [class]="'line series-' + index" />
          @for (point of variant.validationHistory; track point.step) {
            <circle
              [attr.cx]="x(point.step)"
              [attr.cy]="y(point.evaluationLoss)"
              r="4"
              [class]="'dot series-' + index"
            >
              <title>{{ variant.name }} — step {{ point.step }}: {{ fixed(point.evaluationLoss, 6) }}</title>
            </circle>
          }
        }
      </svg>

      <details>
        <summary>Open exact validation values</summary>
        <div class="table-scroll" tabindex="0" aria-label="Validation loss values">
          <table>
            <thead>
              <tr><th scope="col">Step</th>@for (variant of variants; track variant.id) {<th scope="col">{{ variant.name }}</th>}</tr>
            </thead>
            <tbody>
              @for (step of stepTicks(); track step) {
                <tr>
                  <th scope="row">{{ step }}</th>
                  @for (variant of variants; track variant.id) {
                    <td>{{ lossAt(variant, step) }}</td>
                  }
                </tr>
              }
            </tbody>
          </table>
        </div>
      </details>
    </section>
  `,
})
export class TrainingLossChartComponent {
  @Input({ required: true }) variants: readonly ExperimentVariantReport[] = [];

  stepTicks(): readonly number[] {
    return this.variants[0]?.validationHistory.map((point) => point.step) ?? [];
  }

  yTicks(): readonly number[] {
    return [0.44, 0.48, 0.52, 0.56, 0.6];
  }

  x(step: number): number {
    return 58 + ((step - 125) / 875) * 642;
  }

  y(loss: number): number {
    return 218 - ((loss - 0.42) / 0.2) * 185;
  }

  points(variant: ExperimentVariantReport): string {
    return variant.validationHistory
      .map((point) => `${this.x(point.step)},${this.y(point.evaluationLoss)}`)
      .join(' ');
  }

  lossAt(variant: ExperimentVariantReport, step: number): string {
    const value = variant.validationHistory.find((point) => point.step === step)?.evaluationLoss;
    return value === undefined ? '—' : this.fixed(value, 6);
  }

  fixed(value: number, digits: number): string {
    return value.toFixed(digits);
  }
}
