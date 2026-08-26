import '@angular/compiler';

import { describe, expect, it } from 'vitest';

import { BenchmarkResultsComponent } from './benchmark-results.component';

describe('BenchmarkResultsComponent', () => {
  it('formats deployment metrics with stable units and precision', () => {
    const component = new BenchmarkResultsComponent();

    expect(component.formatMilliseconds(1273.603)).toBe('1,273.603 ms');
    expect(component.formatRatio(0.217205)).toBe('0.217205');
    expect(component.formatMemory(766.098)).toBe('766.098 MB');
    expect(component.formatMilliseconds(null)).toBe('—');
  });
});
