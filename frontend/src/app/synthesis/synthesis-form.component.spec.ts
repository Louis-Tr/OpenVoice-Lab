import '@angular/compiler';

import { describe, expect, it, vi } from 'vitest';

import { SynthesisFormComponent } from './synthesis-form.component';

describe('SynthesisFormComponent', () => {
  it('defaults both text-processing options on and keeps them while disabled', () => {
    const component = new SynthesisFormComponent();

    expect(component.sanitizeText).toBe(true);
    expect(component.normalizeText).toBe(true);
    component.disabled = true;

    expect(component.disabled).toBe(true);
    expect(component.sanitizeText).toBe(true);
    expect(component.normalizeText).toBe(true);
  });

  it('emits sanitizer and normalizer changes independently', () => {
    const component = new SynthesisFormComponent();
    const sanitizerListener = vi.fn();
    const normalizerListener = vi.fn();
    component.sanitizeTextChange.subscribe(sanitizerListener);
    component.normalizeTextChange.subscribe(normalizerListener);

    component.onSanitizeTextChange({
      target: { checked: false },
    } as unknown as Event);
    component.onNormalizeTextChange({
      target: { checked: false },
    } as unknown as Event);

    expect(sanitizerListener).toHaveBeenCalledWith(false);
    expect(normalizerListener).toHaveBeenCalledWith(false);
    expect(sanitizerListener).toHaveBeenCalledTimes(1);
    expect(normalizerListener).toHaveBeenCalledTimes(1);
  });
});
