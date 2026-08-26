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
  readonly name: string;
  readonly precision: 'FP32' | 'INT8';
  readonly modelVariant: 'fp32' | 'quantized';
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

export interface BenchmarkConfig {
  readonly corpusVersion: string;
  readonly corpusSha256: string;
  readonly testCaseCount: number;
  readonly modelCount: number;
  readonly totalEvaluations: number;
  readonly modelIds: readonly string[];
  readonly defaultVoiceId: string;
}

export interface BenchmarkJobStatus {
  readonly benchmarkId: string;
  readonly status: 'pending' | 'running' | 'completed' | 'failed';
  readonly testCaseCount: number;
  readonly modelCount: number;
  readonly totalEvaluations: number;
  readonly completedEvaluations: number;
  readonly progressPercent: number;
  readonly result: BenchmarkResult | null;
  readonly error: string | null;
}

@Injectable({ providedIn: 'root' })
export class BenchmarkApiService {
  constructor(
    private readonly http: HttpClient,
    @Inject(API_BASE_URL) private readonly apiBaseUrl: string,
  ) {}

  getConfig(): Observable<BenchmarkConfig> {
    return this.http.get<BenchmarkConfig>(`${this.apiBaseUrl}/benchmarks/config`);
  }

  runBenchmark(request: BenchmarkRequest): Observable<BenchmarkJobStatus> {
    return this.http.post<BenchmarkJobStatus>(`${this.apiBaseUrl}/benchmarks`, request);
  }

  getLatestBenchmark(): Observable<BenchmarkJobStatus> {
    return this.http.get<BenchmarkJobStatus>(`${this.apiBaseUrl}/benchmarks/latest`);
  }

  getBenchmark(benchmarkId: string): Observable<BenchmarkJobStatus> {
    return this.http.get<BenchmarkJobStatus>(
      `${this.apiBaseUrl}/benchmarks/${encodeURIComponent(benchmarkId)}`,
    );
  }
}
