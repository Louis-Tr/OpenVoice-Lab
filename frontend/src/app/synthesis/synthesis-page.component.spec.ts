import '@angular/compiler';

import { HttpErrorResponse } from '@angular/common/http';
import { Subject, of, throwError } from 'rxjs';
import { describe, expect, it, vi } from 'vitest';

import { SynthesisApiService } from '../api/synthesis-api.service';
import { ModelSummary, SynthesisResult } from './synthesis.types';
import { SynthesisPageComponent } from './synthesis-page.component';

const model: ModelSummary = {
  id: 'kokoro-fp32',
  name: 'Kokoro',
  precision: 'FP32',
  variant: 'fp32',
  voices: ['voice-one'],
  modelVersion: '1.0',
  runtime: 'local-runtime',
  hosting: 'self-hosted',
  externalInferenceApis: [],
  available: true,
};

const quantizedModel: ModelSummary = {
  ...model,
  id: 'kokoro-q8',
  precision: 'INT8',
  variant: 'quantized',
};

const metrics = {
  modelLoadMs: 0,
  inferenceMs: 412,
  audioDurationMs: 4310,
  realTimeFactor: 0.095592,
  memoryMb: 715,
  warm: true,
  modelVariant: 'fp32' as const,
};

function createApi(overrides: Partial<SynthesisApiService> = {}): SynthesisApiService {
  return {
    listModels: vi.fn(() => of([model, quantizedModel])),
    synthesize: vi.fn(() =>
      of({
        status: 'ok',
        model: 'kokoro-fp32',
        text: 'Hello',
        audioUrl: '/audio/hello.wav',
        metrics,
      } satisfies SynthesisResult),
    ),
    ...overrides,
  } as unknown as SynthesisApiService;
}

describe('SynthesisPageComponent', () => {
  it('loads available models for a fresh session', () => {
    const component = new SynthesisPageComponent(createApi());

    component.ngOnInit();

    expect(component.modelState()).toBe('ready');
    expect(component.selectedModelId()).toBe('kokoro-fp32');
    expect(component.selectedVoiceId()).toBe('voice-one');
  });

  it('rejects empty input without calling synthesis', () => {
    const api = createApi();
    const component = new SynthesisPageComponent(api);
    component.ngOnInit();

    component.submit();

    expect(component.textError()).toBe('Enter text before generating speech.');
    expect(api.synthesize).not.toHaveBeenCalled();
  });

  it('keeps the loading state active until synthesis completes', () => {
    const response = new Subject<SynthesisResult>();
    const api = createApi({ synthesize: vi.fn(() => response.asObservable()) });
    const component = new SynthesisPageComponent(api);
    component.ngOnInit();
    component.setText('Generate this');

    component.submit();
    expect(component.isSubmitting()).toBe(true);

    response.next({
      status: 'ok',
      model: 'kokoro-fp32',
      text: 'Generate this',
      audioUrl: '/audio/generated.wav',
      metrics,
    });
    response.complete();

    expect(component.isSubmitting()).toBe(false);
    expect(component.result()?.audioUrl).toBe('/audio/generated.wav');
    expect(component.result()?.metrics.realTimeFactor).toBeCloseTo(
      metrics.inferenceMs / metrics.audioDurationMs,
      5,
    );
  });

  it('submits the selected registry configuration without frontend variant logic', () => {
    const api = createApi();
    const component = new SynthesisPageComponent(api);
    component.ngOnInit();
    component.setModelSelection({ modelId: 'kokoro-q8' });
    component.setText('Run the quantized configuration');

    component.submit();

    expect(api.synthesize).toHaveBeenCalledWith({
      text: 'Run the quantized configuration',
      modelId: 'kokoro-q8',
      voiceId: 'voice-one',
    });
  });

  it('gives a recovery path when the backend is unavailable', () => {
    const api = createApi({
      listModels: vi.fn(() =>
        throwError(() => new HttpErrorResponse({ status: 0, statusText: 'Unknown Error' })),
      ),
    });
    const component = new SynthesisPageComponent(api);

    component.ngOnInit();

    expect(component.modelState()).toBe('error');
    expect(component.requestError()).toContain('Backend unavailable');
    expect(component.requestError()).toContain('retry');
  });

  it('reports inference failure without fabricating an audio result', () => {
    const api = createApi({
      synthesize: vi.fn(() =>
        throwError(() => new HttpErrorResponse({ status: 500, statusText: 'Error' })),
      ),
    });
    const component = new SynthesisPageComponent(api);
    component.ngOnInit();
    component.setText('This request fails');

    component.submit();

    expect(component.isSubmitting()).toBe(false);
    expect(component.result()).toBeNull();
    expect(component.requestError()).toContain('Inference failed');
  });
});
