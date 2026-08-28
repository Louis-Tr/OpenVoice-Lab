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
          <p>{{ scaleDescription() }}</p>
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
          Line chart showing validation loss for all four SpeechT5 V1 approaches at steps 25 through 250. {{ scaleDescription() }}
        </desc>
        @if (usesBrokenAxis()) {
          <rect x="58" y="31" width="642" height="41" rx="4" class="plot-band outlier-band" />
          <rect x="58" y="102" width="642" height="116" rx="4" class="plot-band detail-band" />
          <text x="700" y="20" text-anchor="end" class="band-caption outlier-caption">V1D OUTLIER · COMPRESSED 26%</text>
          <text x="700" y="95" text-anchor="end" class="band-caption">V1A–V1C DETAIL · EXPANDED 74%</text>
        }
        @for (tick of yTicks(); track tick) {
          <line x1="58" x2="700" [attr.y1]="y(tick)" [attr.y2]="y(tick)" class="grid" />
          <text x="50" [attr.y]="y(tick) + 4" text-anchor="end">{{ axisLabel(tick) }}</text>
        }
        @if (usesBrokenAxis()) {
          <path d="M 51 80 L 64 86 M 51 88 L 64 94" class="axis-break" />
          <text x="70" y="90" class="break-caption">AXIS BREAK</text>
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

  private readonly upperBand = { top: 31, bottom: 72 };
  private readonly lowerBand = { top: 102, bottom: 218 };

  stepTicks(): readonly number[] {
    return [...new Set(this.variants.flatMap((variant) => variant.validationHistory.map((point) => point.step)))].sort((a, b) => a - b);
  }

  yTicks(): readonly number[] {
    const scale = this.lossScale();
    if (!scale.upper) return this.niceTicks(scale.lower, 5);
    return [...this.niceTicks(scale.upper, 3), ...this.niceTicks(scale.lower, 5)];
  }

  x(step: number): number {
    const steps = this.stepTicks();
    const minimum = steps[0] ?? 0;
    const maximum = steps.at(-1) ?? minimum + 1;
    return 58 + ((step - minimum) / Math.max(1, maximum - minimum)) * 642;
  }

  y(loss: number): number {
    const scale = this.lossScale();
    if (scale.upper && loss >= scale.upper.minimum) {
      return this.mapToBand(loss, scale.upper, this.upperBand);
    }
    return this.mapToBand(loss, scale.lower, scale.upper ? this.lowerBand : { top: 31, bottom: 218 });
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

  axisLabel(value: number): string {
    return value < 1 ? value.toFixed(3) : value.toFixed(2);
  }

  usesBrokenAxis(): boolean {
    return this.lossScale().upper !== null;
  }

  scaleDescription(): string {
    const scale = this.lossScale();
    if (!scale.upper) return 'Ten shared evaluation points on a linear loss axis. Lower is better.';
    return `Broken y-axis: 74% of the plot expands V1A–V1C (${this.fixed(scale.lower.minimum, 2)}–${this.fixed(scale.lower.maximum, 2)}); a compact upper band retains the V1D outlier (${this.fixed(scale.upper.minimum, 2)}–${this.fixed(scale.upper.maximum, 2)}). Lower is better.`;
  }

  private lossScale(): LossScale {
    const losses = this.variants
      .flatMap((variant) => variant.validationHistory.map((point) => point.evaluationLoss))
      .filter((value) => Number.isFinite(value) && value > 0)
      .sort((left, right) => left - right);
    if (losses.length === 0) return { lower: { minimum: 0.1, maximum: 1 }, upper: null };

    let breakIndex = -1;
    let largestRatio = 1;
    for (let index = 0; index < losses.length - 1; index += 1) {
      const ratio = losses[index + 1] / losses[index];
      if (ratio > largestRatio) {
        largestRatio = ratio;
        breakIndex = index;
      }
    }

    if (breakIndex < 0 || largestRatio < 2.5) {
      return { lower: this.paddedBounds(losses), upper: null };
    }

    return {
      lower: this.paddedBounds(losses.slice(0, breakIndex + 1)),
      upper: this.paddedBounds(losses.slice(breakIndex + 1)),
    };
  }

  private paddedBounds(values: readonly number[]): LossBand {
    const minimum = Math.min(...values);
    const maximum = Math.max(...values);
    const padding = Math.max((maximum - minimum) * 0.08, maximum * 0.005, 0.001);
    return { minimum: Math.max(0, minimum - padding), maximum: maximum + padding };
  }

  private mapToBand(loss: number, scale: LossBand, band: { readonly top: number; readonly bottom: number }): number {
    const position = (loss - scale.minimum) / Math.max(0.000001, scale.maximum - scale.minimum);
    return band.bottom - Math.min(1, Math.max(0, position)) * (band.bottom - band.top);
  }

  private niceTicks(band: LossBand, targetCount: number): readonly number[] {
    const rawStep = (band.maximum - band.minimum) / Math.max(1, targetCount - 1);
    const magnitude = 10 ** Math.floor(Math.log10(rawStep));
    const normalized = rawStep / magnitude;
    const multiplier = [1, 2, 2.5, 5, 10].reduce((best, candidate) =>
      Math.abs(candidate - normalized) < Math.abs(best - normalized) ? candidate : best,
    );
    const step = multiplier * magnitude;
    const first = Math.ceil(band.minimum / step) * step;
    const last = Math.floor(band.maximum / step) * step;
    const ticks: number[] = [];
    for (let value = first; value <= last + step / 100; value += step) ticks.push(Number(value.toPrecision(12)));
    return ticks.length > 0 ? ticks : [band.minimum, band.maximum];
  }
}

interface LossBand {
  readonly minimum: number;
  readonly maximum: number;
}

interface LossScale {
  readonly lower: LossBand;
  readonly upper: LossBand | null;
}
