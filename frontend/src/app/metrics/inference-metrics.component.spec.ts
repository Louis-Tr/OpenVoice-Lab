import '@angular/compiler';

import { describe, expect, it } from 'vitest';

import { InferenceMetricsComponent } from './inference-metrics.component';

describe('InferenceMetricsComponent', () => {
  it('formats measurements without changing their underlying values', () => {
    const component = new InferenceMetricsComponent();

    expect(component.formatMilliseconds(412.49)).toBe('412');
    expect(component.formatMemory(714.6)).toBe('715');
    expect(component.formatRtf(0.09563)).toBe('0.096');
  });
});
