import '@angular/compiler';

import { HttpErrorResponse } from '@angular/common/http';
import { Subject, of, throwError } from 'rxjs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { SynthesisApiService } from '../api/synthesis-api.service';
import { ModelSummary, SynthesisJob, SynthesisRequest } from './synthesis.types';
import { SynthesisPageComponent } from './synthesis-page.component';
import { SynthesisSession, SynthesisSessionStore } from './synthesis-session.store';

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
  unavailableReason: null,
  description: 'Full precision local model.',
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


const request: SynthesisRequest = {
  text: 'Hello', modelId: model.id, voiceId: 'voice-one',
  sanitizeText: true, normalizeText: true,
};

function job(status: SynthesisJob['status'] = 'completed', input = request): SynthesisJob {
  return {
    jobId: 'job-one', status, request: input,
    createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(),
    completedAt: status === 'completed' ? new Date().toISOString() : null,
    expiresAt: null, error: null,
    result: status === 'completed' ? {
      status: 'ok', model: input.modelId, text: input.text,
      normalizedText: input.text, audioUrl: '/audio/hello.wav', metrics,
    } : null,
  };
}

function memoryStorage(): Storage {
  const entries = new Map<string, string>();
  return {
    get length() { return entries.size; },
    key: (index) => [...entries.keys()][index] ?? null,
    getItem: (key) => entries.get(key) ?? null,
    setItem: (key, value) => { entries.set(key, value); },
    removeItem: (key) => { entries.delete(key); },
    clear: () => entries.clear(),
  };
}

function session(overrides: Partial<SynthesisSession> = {}): SynthesisSession {
  return {
    version: 1, key: 'original-key', jobId: null, request,
    createdAt: Date.now(), stoppedReason: null, ...overrides,
  };
}

const components: SynthesisPageComponent[] = [];
function createComponent(api: SynthesisApiService, storage = memoryStorage()): SynthesisPageComponent {
  const component = new SynthesisPageComponent(api, new SynthesisSessionStore(storage, '/api'));
  components.push(component);
  return component;
}

function createApi(overrides: Partial<SynthesisApiService> = {}): SynthesisApiService {
  return {
    listModels: vi.fn(() => of([model, quantizedModel])),
    startJob: vi.fn((input: SynthesisRequest) => of(job('completed', input))),
    getJob: vi.fn(() => of(job())),
    ...overrides,
  } as unknown as SynthesisApiService;
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => {
  for (const component of components.splice(0)) component.ngOnDestroy();
  vi.useRealTimers();
});

describe('SynthesisPageComponent', () => {
  it('persists the exact submission before POST and blocks double clicks', () => {
    const storage = memoryStorage();
    const store = new SynthesisSessionStore(storage, '/api');
    const pending = new Subject<SynthesisJob>();
    const api = createApi({ startJob: vi.fn((input, key) => {
      expect(store.load()).toMatchObject({ key, request: input, jobId: null });
      return pending;
    }) });
    const component = createComponent(api, storage);
    component.ngOnInit();
    component.setText(' Save this exact request ');
    component.sanitizeText.set(false);
    component.submit();
    component.submit();
    expect(api.startJob).toHaveBeenCalledTimes(1);
    expect(store.load()?.request).toEqual({ ...request, text: 'Save this exact request', sanitizeText: false });
  });

  it('restores a running job, preserves selection through delayed model loading, and stops polling', () => {
    const storage = memoryStorage();
    const store = new SynthesisSessionStore(storage, '/api');
    const input = { ...request, modelId: quantizedModel.id, voiceId: 'voice-two', normalizeText: false };
    store.save(session({ request: input, jobId: 'job-one' }));
    const catalog = new Subject<readonly ModelSummary[]>();
    const api = createApi({
      listModels: vi.fn(() => catalog),
      getJob: vi.fn().mockReturnValueOnce(of(job('generating', input)))
        .mockReturnValue(of(job('completed', input))),
    });
    const component = createComponent(api, storage);
    component.ngOnInit();
    expect(component.isSubmitting()).toBe(true);
    expect(component.text()).toBe(input.text);
    expect(component.normalizeText()).toBe(false);
    catalog.next([model, { ...quantizedModel, voices: ['voice-one', 'voice-two'] }]);
    expect(component.selectedModelId()).toBe(quantizedModel.id);
    expect(component.selectedVoiceId()).toBe('voice-two');
    vi.advanceTimersByTime(0);
    expect(component.generationStatus()).toBe('Generating your audio…');
    vi.advanceTimersByTime(2000);
    expect(component.isSubmitting()).toBe(false);
    expect(component.result()?.metrics).toEqual(metrics);
    expect(component.result()?.audioUrl).toBe('/audio/hello.wav');
    vi.advanceTimersByTime(6000);
    expect(api.getJob).toHaveBeenCalledTimes(2);
    expect(api.startJob).not.toHaveBeenCalled();
  });

  it('refresh after a lost POST response reuses the saved key and body', () => {
    const storage = memoryStorage();
    const firstApi = createApi({ startJob: vi.fn(() => new Subject<SynthesisJob>()) });
    const first = createComponent(firstApi, storage);
    first.ngOnInit();
    first.setText('An unfinished request');
    first.submit();
    const saved = new SynthesisSessionStore(storage, '/api').load()!;
    first.ngOnDestroy();
    const refreshedApi = createApi();
    const refreshed = createComponent(refreshedApi, storage);
    refreshed.ngOnInit();
    expect(refreshedApi.startJob).toHaveBeenCalledExactlyOnceWith(saved.request, saved.key);
    expect(refreshed.result()?.audioUrl).toBe('/audio/hello.wav');
  });

  it('refresh after an acknowledged POST uses GET and restores the completed result', () => {
    const storage = memoryStorage();
    const firstApi = createApi({ startJob: vi.fn(() => of(job('generating'))) });
    const first = createComponent(firstApi, storage);
    first.ngOnInit();
    first.setText('Hello');
    first.submit();
    first.ngOnDestroy();
    vi.advanceTimersByTime(4000);
    expect(firstApi.getJob).not.toHaveBeenCalled();
    const refreshedApi = createApi();
    const refreshed = createComponent(refreshedApi, storage);
    refreshed.ngOnInit();
    vi.advanceTimersByTime(0);
    expect(refreshedApi.getJob).toHaveBeenCalledExactlyOnceWith('job-one');
    expect(refreshedApi.startJob).not.toHaveBeenCalled();
    expect(refreshed.result()).toEqual(job().result);
    expect(refreshed.isSubmitting()).toBe(false);
  });

  it.each([0, 502, 503])('keeps an ambiguous POST (%s) recoverable with the same key', (status) => {
    const storage = memoryStorage();
    const api = createApi({ startJob: vi.fn()
      .mockReturnValueOnce(throwError(() => new HttpErrorResponse({ status })))
      .mockReturnValue(of(job())),
    });
    const component = createComponent(api, storage);
    component.ngOnInit();
    component.setText('Hello');
    component.submit();
    expect(component.needsReconnect()).toBe(true);
    expect(component.isSubmitting()).toBe(true);
    component.submit();
    expect(api.startJob).toHaveBeenCalledTimes(1);
    const saved = new SynthesisSessionStore(storage, '/api').load()!;
    component.reconnect();
    expect(api.startJob).toHaveBeenLastCalledWith(saved.request, saved.key);
    expect(component.needsReconnect()).toBe(false);
    expect(component.isSubmitting()).toBe(false);
  });

  it('a polling failure preserves the known job and Check generation never POSTs', () => {
    const storage = memoryStorage();
    new SynthesisSessionStore(storage, '/api').save(session({ jobId: 'job-one' }));
    const api = createApi({ getJob: vi.fn()
      .mockReturnValueOnce(throwError(() => new HttpErrorResponse({ status: 503 })))
      .mockReturnValue(of(job())),
    });
    const component = createComponent(api, storage);
    component.ngOnInit();
    vi.advanceTimersByTime(0);
    expect(component.needsReconnect()).toBe(true);
    expect(component.isSubmitting()).toBe(true);
    component.reconnect();
    vi.advanceTimersByTime(0);
    expect(api.getJob).toHaveBeenCalledTimes(2);
    expect(api.startJob).not.toHaveBeenCalled();
    expect(component.result()).not.toBeNull();
  });

  it('does not cancel slow status requests or overlap polling', () => {
    const storage = memoryStorage();
    new SynthesisSessionStore(storage, '/api').save(session({ jobId: 'job-one' }));
    const slow = new Subject<SynthesisJob>();
    const api = createApi({ getJob: vi.fn(() => slow) });
    const component = createComponent(api, storage);
    component.ngOnInit();
    vi.advanceTimersByTime(8000);
    expect(api.getJob).toHaveBeenCalledTimes(1);
    slow.next(job());
    expect(component.result()).not.toBeNull();
    expect(slow.observed).toBe(false);
    vi.advanceTimersByTime(8000);
    expect(api.getJob).toHaveBeenCalledTimes(1);
  });

  it.each([404, 410])('restores text but does not recreate a missing job (%s)', (status) => {
    const storage = memoryStorage();
    new SynthesisSessionStore(storage, '/api').save(session({ jobId: 'job-one' }));
    const api = createApi({ getJob: vi.fn(() => throwError(() => new HttpErrorResponse({ status }))) });
    const component = createComponent(api, storage);
    component.ngOnInit();
    vi.advanceTimersByTime(0);
    expect(component.text()).toBe('Hello');
    expect(component.requestError()).toContain('backend restarted');
    expect(component.isSubmitting()).toBe(false);
    expect(api.startJob).not.toHaveBeenCalled();
    const refreshed = createComponent(api, storage);
    refreshed.ngOnInit();
    vi.advanceTimersByTime(0);
    expect(api.getJob).toHaveBeenCalledTimes(1);
    expect(refreshed.requestError()).toContain('backend restarted');
  });

  it.each([409, 422, 503])('retains definitive rejection (%s) and uses a new key only on Generate', (status) => {
    const storage = memoryStorage();
    const api = createApi({ startJob: vi.fn(() => throwError(() =>
      new HttpErrorResponse({ status, error: { detail: 'Request rejected' } }),
    )) });
    const component = createComponent(api, storage);
    component.ngOnInit();
    component.setText('Hello');
    component.submit();
    const firstKey = new SynthesisSessionStore(storage, '/api').load()!.key;
    expect(component.isSubmitting()).toBe(false);
    const refreshed = createComponent(api, storage);
    refreshed.ngOnInit();
    expect(refreshed.requestError()).toBe('Request rejected');
    expect(api.startJob).toHaveBeenCalledTimes(1);
    refreshed.submit();
    expect(new SynthesisSessionStore(storage, '/api').load()!.key).not.toBe(firstKey);
    expect(api.startJob).toHaveBeenCalledTimes(2);
  });

  it('restores failed jobs without resubmitting', () => {
    const storage = memoryStorage();
    new SynthesisSessionStore(storage, '/api').save(session({ jobId: 'job-one' }));
    const api = createApi({ getJob: vi.fn(() => of({
      ...job('failed'), error: { statusCode: 500, detail: 'Storage failed' },
    })) });
    const component = createComponent(api, storage);
    component.ngOnInit();
    vi.advanceTimersByTime(0);
    expect(component.requestError()).toBe('Storage failed');
    expect(component.result()).toBeNull();
    expect(component.isSubmitting()).toBe(false);
    expect(api.startJob).not.toHaveBeenCalled();
  });

  it('does not blindly replay unacknowledged submissions older than 24 hours', () => {
    const storage = memoryStorage();
    new SynthesisSessionStore(storage, '/api').save(session({ createdAt: Date.now() - 86_400_001 }));
    const api = createApi();
    const component = createComponent(api, storage);
    component.ngOnInit();
    expect(component.requestError()).toContain('too old');
    expect(api.startJob).not.toHaveBeenCalled();
    expect(component.text()).toBe('Hello');
  });

  it('handles corrupt saved state and refused browser storage without sending a request', () => {
    const storage = memoryStorage();
    storage.setItem('openvoice.synthesis.v1:/api', '{bad json');
    const api = createApi();
    const component = createComponent(api, storage);
    component.ngOnInit();
    expect(component.text()).toBe('');
    storage.setItem = () => { throw new Error('Quota exceeded'); };
    component.setText('Hello');
    component.submit();
    expect(component.isSubmitting()).toBe(false);
    expect(component.requestError()).toContain('site storage');
    expect(api.startJob).not.toHaveBeenCalled();
  });

  it('restores a result even when the model catalog is unavailable', () => {
    const storage = memoryStorage();
    new SynthesisSessionStore(storage, '/api').save(session({ jobId: 'job-one' }));
    const api = createApi({ listModels: vi.fn(() => throwError(() => new HttpErrorResponse({ status: 0 }))) });
    const component = createComponent(api, storage);
    component.ngOnInit();
    vi.advanceTimersByTime(0);
    expect(component.result()).not.toBeNull();
    expect(api.startJob).not.toHaveBeenCalled();
  });

  it('isolates saved sessions by API and rejects malformed persisted request options', () => {
    const storage = memoryStorage();
    const first = new SynthesisSessionStore(storage, '/api');
    const other = new SynthesisSessionStore(storage, 'https://other.test/api');
    first.save(session());
    expect(other.load()).toBeNull();
    storage.setItem('openvoice.synthesis.v1:/api', JSON.stringify({ ...session(), request: { text: 'Hello' } }));
    expect(first.load()).toBeNull();
  });

  it('allows 5000 characters for SpeechT5 and preserves an overlength draft', () => {
    const speechT5: ModelSummary = {
      ...model,
      id: 'speecht5-pretrained',
      name: 'SpeechT5',
      maxInputCharacters: 5000,
      maxInputTokens: 600,
    };
    const api = createApi({ listModels: vi.fn(() => of([model, speechT5])) });
    const component = createComponent(api);
    component.ngOnInit();
    component.setText('a'.repeat(5000));
    expect(component.inputLimit()).toBe(5000);
    expect(component.inputLimitError()).toBe('');

    component.setModelSelection({ modelId: speechT5.id });
    expect(component.inputLimit()).toBe(5000);
    expect(component.inputLimitError()).toBe('');
    component.setText('a'.repeat(5001));
    expect(component.inputLimitError()).toContain('SpeechT5 accepts up to 5000');
    expect(component.text()).toHaveLength(5001);
    component.submit();
    expect(api.startJob).not.toHaveBeenCalled();

    component.setText('a'.repeat(5000));
    expect(component.inputLimitError()).toBe('');
    component.submit();
    expect(api.startJob).toHaveBeenCalledTimes(1);

    component.setModelSelection({ modelId: model.id });
    expect(component.inputLimit()).toBe(5000);
  });

  it('shows the backend limit message when cleanup expands the input', () => {
    const detail = 'Processed text exceeds 5000 characters.';
    const api = createApi({
      startJob: vi.fn(() => throwError(() => new HttpErrorResponse({
        status: 422, error: { detail },
      }))),
    });
    const component = createComponent(api);
    component.ngOnInit();
    component.setText('A short input.');
    component.submit();
    expect(component.requestError()).toBe(detail);
  });

  it('loads available models for a fresh session', () => {
    const component = createComponent(createApi());

    component.ngOnInit();

    expect(component.modelState()).toBe('ready');
    expect(component.selectedModelId()).toBe('kokoro-fp32');
    expect(component.selectedVoiceId()).toBe('voice-one');
  });

  it('keeps unavailable catalog entries visible while selecting the first ready model', () => {
    const unavailable: ModelSummary = {
      ...model,
      id: 'audio8-0.6b',
      name: 'Audio8 0.6B',
      precision: 'BF16',
      variant: 'audio8',
      available: false,
      unavailableReason: 'Runtime not provisioned.',
    };
    const component = createComponent(
      createApi({ listModels: vi.fn(() => of([unavailable, model])) }),
    );

    component.ngOnInit();

    expect(component.models()).toEqual([unavailable, model]);
    expect(component.selectedModelId()).toBe('kokoro-fp32');
  });

  it('rejects empty input without calling synthesis', () => {
    const api = createApi();
    const component = createComponent(api);
    component.ngOnInit();

    component.submit();

    expect(component.textError()).toBe('Enter text before generating speech.');
    expect(api.startJob).not.toHaveBeenCalled();
  });

  it('keeps the loading state active until synthesis completes', () => {
    const response = new Subject<SynthesisJob>();
    const api = createApi({ startJob: vi.fn(() => response.asObservable()) });
    const component = createComponent(api);
    component.ngOnInit();
    component.setText('Generate this');

    component.submit();
    expect(component.isSubmitting()).toBe(true);

    response.next({
      ...job('completed', { ...request, text: 'Generate this' }),
      result: {
        status: 'ok', model: 'kokoro-fp32', text: 'Generate this',
        normalizedText: 'Generate this', audioUrl: '/audio/generated.wav', metrics,
      },
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
    const component = createComponent(api);
    component.ngOnInit();
    component.setModelSelection({ modelId: 'kokoro-q8' });
    component.setText('Run the quantized configuration');

    component.submit();

    expect(api.startJob).toHaveBeenCalledWith({
      text: 'Run the quantized configuration',
      modelId: 'kokoro-q8',
      voiceId: 'voice-one',
      sanitizeText: true,
      normalizeText: true,
    }, expect.any(String));
  });

  it('defaults both processing options on and sends independent option states', () => {
    const api = createApi();
    const component = createComponent(api);
    component.ngOnInit();
    component.setText('Keep ./ -- $25 exactly.');

    expect(component.sanitizeText()).toBe(true);
    expect(component.normalizeText()).toBe(true);
    component.sanitizeText.set(false);
    component.submit();

    expect(api.startJob).toHaveBeenCalledWith({
      text: 'Keep ./ -- $25 exactly.',
      modelId: 'kokoro-fp32',
      voiceId: 'voice-one',
      sanitizeText: false,
      normalizeText: true,
    }, expect.any(String));
  });

  it.each([
    [true, true],
    [true, false],
    [false, true],
    [false, false],
  ])(
    'sends sanitizer=%s and normalizer=%s without coupling them',
    (sanitizeText, normalizeText) => {
      const api = createApi();
      const component = createComponent(api);
      component.ngOnInit();
      component.setText('Process this text');
      component.sanitizeText.set(sanitizeText);
      component.normalizeText.set(normalizeText);

      component.submit();

      expect(api.startJob).toHaveBeenCalledWith(
        expect.objectContaining({ sanitizeText, normalizeText }), expect.any(String),
      );
    },
  );

  it('shows the inference input after synthesis whether or not preprocessing changed it', () => {
    const component = createComponent(createApi());

    expect(component.processedTextPreview()).toBeNull();

    component.result.set({
      status: 'ok',
      model: 'kokoro-fp32',
      text: 'Save 15% today.',
      normalizedText: 'Save 15 percent today.',
      audioUrl: '/audio/normalized.wav',
      metrics,
    });
    expect(component.processedTextPreview()).toBe('Save 15 percent today.');

    component.result.set({
      status: 'ok',
      model: 'kokoro-fp32',
      text: 'Natural speech.',
      normalizedText: 'Natural speech.',
      audioUrl: '/audio/unchanged.wav',
      metrics,
    });
    expect(component.processedTextPreview()).toBe('Natural speech.');
  });

  it('gives a recovery path when the backend is unavailable', () => {
    const api = createApi({
      listModels: vi.fn(() =>
        throwError(() => new HttpErrorResponse({ status: 0, statusText: 'Unknown Error' })),
      ),
    });
    const component = createComponent(api);

    component.ngOnInit();

    expect(component.modelState()).toBe('error');
    expect(component.requestError()).toContain('Backend unavailable');
    expect(component.requestError()).toContain('retry');
  });

  it('reports inference failure without fabricating an audio result', () => {
    const api = createApi({
      startJob: vi.fn(() =>
        of({ ...job('failed'), error: { statusCode: 500, detail: 'Inference failed' } }),
      ),
    });
    const component = createComponent(api);
    component.ngOnInit();
    component.setText('This request fails');

    component.submit();

    expect(component.isSubmitting()).toBe(false);
    expect(component.result()).toBeNull();
    expect(component.requestError()).toContain('Inference failed');
  });
});
