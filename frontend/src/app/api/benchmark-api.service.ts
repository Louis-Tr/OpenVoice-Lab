import { HttpClient } from '@angular/common/http';
import { Inject, Injectable } from '@angular/core';
import { Observable } from 'rxjs';

import { API_BASE_URL } from '../core/api-base-url.token';

export interface BenchmarkRequest {
  readonly modelIds?: readonly string[];
  readonly voiceId: string;
}

export interface BenchmarkAggregate {
  readonly modelId: string;
  readonly precision: 'FP32' | 'INT8';
  readonly totalCases: number;
  readonly successCount: number;
  readonly failureCount: number;
  readonly averageLatencyMs: number | null;
  readonly medianLatencyMs: number | null;
  readonly p95LatencyMs: number | null;
  readonly averageRealTimeFactor: number | null;
  readonly averageMemoryMb: number | null;
  readonly peakMemoryMb: number | null;
}

export interface BenchmarkResult {
  readonly benchmarkId: string;
  readonly status: 'completed' | 'completed_with_failures';
  readonly corpusVersion: string;
  readonly corpusSha256: string;
  readonly aggregates: readonly BenchmarkAggregate[];
  readonly resultFile: string | null;
}

@Injectable({ providedIn: 'root' })
export class BenchmarkApiService {
  constructor(
    private readonly http: HttpClient,
    @Inject(API_BASE_URL) private readonly apiBaseUrl: string,
  ) {}

  runBenchmark(request: BenchmarkRequest): Observable<BenchmarkResult> {
    return this.http.post<BenchmarkResult>(`${this.apiBaseUrl}/benchmarks`, request);
  }
}
