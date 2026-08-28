import '@angular/compiler';

import { describe, expect, it } from 'vitest';

import { ExperimentVariantReport } from './experiment.types';
import { TrainingLossChartComponent } from './training-loss-chart.component';

const variant = (id: string, losses: readonly number[]) => ({
  id,
  validationHistory: losses.map((evaluationLoss, index) => ({
    step: (index + 1) * 25,
    epoch: 0,
    evaluationLoss,
  })),
}) as unknown as ExperimentVariantReport;

describe('TrainingLossChartComponent', () => {
  const component = new TrainingLossChartComponent();
  component.variants = [
    variant('v1a-conservative-full', [0.760, 0.708]),
    variant('v1b-lora', [0.798, 0.706]),
    variant('v1c-gradual-unfreeze', [0.804, 0.749]),
    variant('v1d-reduction-factor-1', [5.811, 5.233]),
  ];

  it('uses an explicit broken axis when one loss series is a large outlier', () => {
    expect(component.usesBrokenAxis()).toBe(true);
    expect(component.scaleDescription()).toContain('74% of the plot expands V1A–V1C');
  });

  it('expands the useful low-loss range while retaining V1D in the upper band', () => {
    expect(Math.abs(component.y(0.71) - component.y(0.78))).toBeGreaterThan(50);
    expect(component.y(0.80)).toBeGreaterThan(100);
    expect(component.y(5.23)).toBeLessThan(73);
  });
});
