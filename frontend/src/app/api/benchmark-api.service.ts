import { HttpClient } from '@angular/common/http';
import { Inject, Injectable } from '@angular/core';
import { Observable } from 'rxjs';

import { API_BASE_URL } from '../core/api-base-url.token';

export interface BenchmarkRequest {
  readonly modelId: string;
  readonly variant: 'fp32' | 'quantized';
}

export interface BenchmarkResult {
  readonly benchmarkId: string;
  readonly status: string;
  readonly aggregates: Readonly<Record<string, number>>;
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

