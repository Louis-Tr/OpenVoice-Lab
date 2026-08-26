import '@angular/compiler';

import { describe, expect, it, vi } from 'vitest';

import { SynthesisFormComponent } from './synthesis-form.component';

describe('SynthesisFormComponent', () => {
  it('defaults text sanitization on and keeps it while the form is disabled', () => {
    const component = new SynthesisFormComponent();

    expect(component.sanitizeText).toBe(true);
    component.disabled = true;

    expect(component.disabled).toBe(true);
    expect(component.sanitizeText).toBe(true);
  });

  it('emits sanitizer changes as an independent form option', () => {
    const component = new SynthesisFormComponent();
    const listener = vi.fn();
    component.sanitizeTextChange.subscribe(listener);

    component.onSanitizeTextChange({
      target: { checked: false },
    } as unknown as Event);

    expect(listener).toHaveBeenCalledWith(false);
  });
});
