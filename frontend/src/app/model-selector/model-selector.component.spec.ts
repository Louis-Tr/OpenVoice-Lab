import '@angular/compiler';

import { describe, expect, it, vi } from 'vitest';

import { ModelSummary } from '../synthesis/synthesis.types';
import { ModelSelectorComponent } from './model-selector.component';

const models: readonly ModelSummary[] = [
  {
    id: 'kokoro-fp32',
    name: 'Kokoro',
    precision: 'FP32',
    variant: 'fp32',
    voices: ['af_heart'],
    modelVersion: '1.0',
    runtime: 'ONNX',
    hosting: 'self-hosted',
    externalInferenceApis: [],
    available: true,
  },
  {
    id: 'kokoro-q8',
    name: 'Kokoro',
    precision: 'INT8',
    variant: 'quantized',
    voices: ['af_heart'],
    modelVersion: '1.0',
    runtime: 'ONNX',
    hosting: 'self-hosted',
    externalInferenceApis: [],
    available: true,
  },
];

describe('ModelSelectorComponent', () => {
  it('builds selectable labels from API metadata', () => {
    const component = new ModelSelectorComponent();
    component.models = models;

    expect(component.choices).toEqual([
      { modelId: 'kokoro-fp32', label: 'Kokoro FP32' },
      { modelId: 'kokoro-q8', label: 'Kokoro INT8' },
    ]);
  });

  it('emits only the stable registry ID selected by the user', () => {
    const component = new ModelSelectorComponent();
    component.models = models;
    const listener = vi.fn();
    component.modelSelectionChange.subscribe(listener);

    component.onModelChange({
      target: { value: 'kokoro-q8' },
    } as unknown as Event);

    expect(listener).toHaveBeenCalledWith({ modelId: 'kokoro-q8' });
  });
});
