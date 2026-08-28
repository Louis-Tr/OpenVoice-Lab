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
    unavailableReason: null,
    description: 'Full precision local model.',
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
    unavailableReason: null,
    description: 'Quantized local model.',
  },
];

describe('ModelSelectorComponent', () => {
  it('builds selectable labels from API metadata', () => {
    const component = new ModelSelectorComponent();
    component.models = models;

    expect(component.choices).toEqual([
      {
        modelId: 'kokoro-fp32',
        label: 'Kokoro FP32',
        precision: 'FP32',
        runtime: 'ONNX',
        available: true,
        unavailableReason: null,
      },
      {
        modelId: 'kokoro-q8',
        label: 'Kokoro INT8',
        precision: 'INT8',
        runtime: 'ONNX',
        available: true,
        unavailableReason: null,
      },
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

  it('explains an unavailable catalog model without emitting a selection', () => {
    const component = new ModelSelectorComponent();
    const listener = vi.fn();
    component.modelSelectionChange.subscribe(listener);
    component.models = [
      ...models,
      {
        ...models[0],
        id: 'audio8-0.6b',
        name: 'Audio8 0.6B',
        precision: 'BF16',
        variant: 'audio8',
        available: false,
        unavailableReason: 'Provision the reviewed Audio8 runtime first.',
      },
    ];

    component.selectChoice(component.choices[2]);

    expect(listener).not.toHaveBeenCalled();
    expect(component.catalogNotice).toContain('Provision the reviewed Audio8 runtime');
  });
});
