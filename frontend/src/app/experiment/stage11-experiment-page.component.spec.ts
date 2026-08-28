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

const v1a = {
  id: 'v1a-conservative-full',
  name: 'V1A Conservative Full',
  evaluation: evaluation(0.8918, 0.2253),
} as unknown as ExperimentVariantReport;

const v1c = {
  id: 'v1c-gradual-unfreeze',
  name: 'V1C Gradual Unfreeze',
  evaluation: evaluation(0.9159, 0.1378),
} as unknown as ExperimentVariantReport;

const report = {
  pretrainedControl: {
    id: 'speecht5-pretrained',
    name: 'SpeechT5 Pretrained',
    evaluation: evaluation(0.9159, 0.111),
  },
  variants: [v1a, v1c],
} as unknown as ExperimentReport;

describe('Stage11ExperimentPageComponent', () => {
  const component = new Stage11ExperimentPageComponent({} as ExperimentApiService);

  it('selects the measured pretrained control as the primary-metric winner', () => {
    expect(component.primaryWinnerId(report)).toBe('speecht5-pretrained');
  });

  it('does not recommend the tied V1C checkpoint over the lower-WER control', () => {
    expect(component.conclusion(v1c, report)).toContain('do not replace the control');
    expect(component.measuredOutcome(v1c, report)).toContain('Matched pretrained');
  });
});
