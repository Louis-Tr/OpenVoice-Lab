import '@angular/compiler';

import { HttpErrorResponse } from '@angular/common/http';
import { of, throwError } from 'rxjs';
import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  BenchmarkApiService,
  BenchmarkConfig,
  BenchmarkJobStatus,
  BenchmarkResult,
} from '../api/benchmark-api.service';
import { BenchmarkPageComponent } from './benchmark-page.component';

const config: BenchmarkConfig = {
  corpusVersion: '1.0.0',
  corpusSha256: 'a'.repeat(64),
  testCaseCount: 8,
  modelCount: 5,
  totalEvaluations: 40,
  modelIds: [
    'kokoro-fp32',
    'kokoro-fp16',
    'kokoro-q8',
    'audio8-0.6b',
    'speecht5-pretrained',
  ],
  modelVoiceIds: {
    'kokoro-fp32': 'af_heart',
    'kokoro-fp16': 'af_heart',
    'kokoro-q8': 'af_heart',
    'audio8-0.6b': 'unconditioned',
    'speecht5-pretrained': 'cmu-slt',
  },
  defaultVoiceId: null,
};

const result: BenchmarkResult = {
  benchmarkId: 'benchmark-test',
  status: 'completed',
  corpusVersion: '1.0.0',
  corpusSha256: 'a'.repeat(64),
  voiceId: null,
  modelIds: ['kokoro-fp32', 'kokoro-q8'],
  modelVoiceIds: {
    'kokoro-fp32': 'af_heart',
    'kokoro-q8': 'af_heart',
  },
  aggregates: [
    {
      modelId: 'kokoro-fp32',
      name: 'Kokoro',
      precision: 'FP32',
      modelVariant: 'fp32',
      totalCases: 8,
      successCount: 8,
      failureCount: 0,
      averageLatencyMs: 1273.603,
      medianLatencyMs: 1019.679,
      p95LatencyMs: 2571.653,
      averageRealTimeFactor: 0.217205,
      averageMemoryMb: 698.113,
      peakMemoryMb: 766.098,
    },
    {
      modelId: 'kokoro-q8',
      name: 'Kokoro',
      precision: 'INT8',
      modelVariant: 'quantized',
      totalCases: 8,
      successCount: 8,
      failureCount: 0,
      averageLatencyMs: 12865.738,
      medianLatencyMs: 10951.395,
      p95LatencyMs: 25304.652,
      averageRealTimeFactor: 2.15127,
      averageMemoryMb: 527.483,
      peakMemoryMb: 601.859,
    },
  ],
  resultFile: 'benchmark-test.json',
};

const pendingJob: BenchmarkJobStatus = {
  benchmarkId: 'benchmark-test',
  status: 'pending',
  testCaseCount: 8,
  modelCount: 2,
  totalEvaluations: 16,
  completedEvaluations: 0,
  progressPercent: 0,
  result: null,
  error: null,
};

const runningJob: BenchmarkJobStatus = {
  ...pendingJob,
  status: 'running',
  completedEvaluations: 8,
  progressPercent: 50,
};

const completedJob: BenchmarkJobStatus = {
  ...pendingJob,
  status: 'completed',
  completedEvaluations: 16,
  progressPercent: 100,
  result,
};

function createApi(overrides: Partial<BenchmarkApiService> = {}): BenchmarkApiService {
  return {
    getConfig: vi.fn(() => of(config)),
    getLatestBenchmark: vi.fn(() =>
      throwError(() => new HttpErrorResponse({ status: 404, statusText: 'Not Found' })),
    ),
    runBenchmark: vi.fn(() => of(pendingJob)),
    getBenchmark: vi.fn(() => of(completedJob)),
    ...overrides,
  } as unknown as BenchmarkApiService;
}

afterEach(() => {
  vi.useRealTimers();
});

describe('BenchmarkPageComponent', () => {
  it('loads the fixed workload contract for a fresh browser session', () => {
    const component = new BenchmarkPageComponent(createApi());

    component.ngOnInit();

    expect(component.configState()).toBe('ready');
    expect(component.config()?.testCaseCount).toBe(8);
    expect(component.config()?.totalEvaluations).toBe(40);
  });

  it('triggers, tracks, and renders a completed comparison', async () => {
    vi.useFakeTimers();
    const getBenchmark = vi
      .fn()
      .mockReturnValueOnce(of(runningJob))
      .mockReturnValueOnce(of(completedJob));
    const api = createApi({ getBenchmark });
    const component = new BenchmarkPageComponent(api);
    component.ngOnInit();

    component.runBenchmark();
    await vi.advanceTimersByTimeAsync(1);

    expect(api.runBenchmark).toHaveBeenCalledWith({
      modelIds: [
        'kokoro-fp32',
        'kokoro-fp16',
        'kokoro-q8',
        'audio8-0.6b',
        'speecht5-pretrained',
      ],
    });
    expect(component.job()?.completedEvaluations).toBe(8);
    expect(component.resultsState()).toBe('loading');

    await vi.advanceTimersByTimeAsync(2_000);

    expect(component.job()?.status).toBe('completed');
    expect(component.job()?.result?.aggregates).toHaveLength(2);
    expect(component.resultsState()).toBe('ready');
    component.ngOnDestroy();
  });

  it('shows the benchmark failure returned by the job', async () => {
    vi.useFakeTimers();
    const failedJob: BenchmarkJobStatus = {
      ...pendingJob,
      status: 'failed',
      error: 'The isolated worker stopped.',
    };
    const component = new BenchmarkPageComponent(
      createApi({ getBenchmark: vi.fn(() => of(failedJob)) }),
    );
    component.ngOnInit();

    component.runBenchmark();
    await vi.advanceTimersByTimeAsync(1);

    expect(component.resultsState()).toBe('error');
    expect(component.requestError()).toBe('The isolated worker stopped.');
    component.ngOnDestroy();
  });

  it('provides a recovery message when the backend is unavailable', () => {
    const component = new BenchmarkPageComponent(
      createApi({
        getConfig: vi.fn(() =>
          throwError(
            () => new HttpErrorResponse({ status: 0, statusText: 'Unknown Error' }),
          ),
        ),
      }),
    );

    component.ngOnInit();

    expect(component.configState()).toBe('error');
    expect(component.requestError()).toContain('Backend unavailable');
  });
});
