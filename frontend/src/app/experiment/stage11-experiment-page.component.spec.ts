import '@angular/compiler';

import { describe, expect, it } from 'vitest';

import { ExperimentApiService } from '../api/experiment-api.service';
import { ExperimentReport, ExperimentVariantReport } from './experiment.types';
import { Stage11ExperimentPageComponent } from './stage11-experiment-page.component';

const evaluation = (domainTermAccuracy: number, wordErrorRate: number) => ({
  caseCount: 662,
  failureCount: 0,
  domainTermsCorrect: 0,
  domainTermsTotal: 416,
  domainTermAccuracy,
  wordErrorRate,
  averageInferenceMs: 1_000,
  averageRealTimeFactor: 0.18,
  peakGpuMemoryMb: 1_000,
  synthesisVerified: true,
});

const v1 = {
  id: 'v1-baseline',
  name: 'V1 Baseline',
  evaluation: evaluation(0.3341, 0.7084),
} as unknown as ExperimentVariantReport;

const v3 = {
  id: 'v3-replay',
  name: 'V3 Replay',
  evaluation: evaluation(0.351, 0.7019),
} as unknown as ExperimentVariantReport;

const report = {
  pretrainedControl: {
    id: 'speecht5-pretrained',
    name: 'SpeechT5 Pretrained',
    evaluation: evaluation(0.9159, 0.111),
  },
  variants: [v1, v3],
} as unknown as ExperimentReport;

describe('Stage11ExperimentPageComponent', () => {
  const component = new Stage11ExperimentPageComponent({} as ExperimentApiService);

  it('selects the measured pretrained control as the primary-metric winner', () => {
    expect(component.primaryWinnerId(report)).toBe('speecht5-pretrained');
  });

  it('does not recommend the best adapted checkpoint over the stronger control', () => {
    expect(component.conclusion(v3, report)).toContain('do not prefer it over pretrained');
    expect(component.conclusion(v3, report)).toContain('56.49 points ahead');
  });
});
