import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Inject, Injectable } from '@angular/core';
import { Observable } from 'rxjs';

import { API_BASE_URL } from '../core/api-base-url.token';
import {
  ModelSummary,
  SynthesisRequest,
  SynthesisJob,
  SynthesisResult,
} from '../synthesis/synthesis.types';

@Injectable({ providedIn: 'root' })
export class SynthesisApiService {
  constructor(
    private readonly http: HttpClient,
    @Inject(API_BASE_URL) private readonly apiBaseUrl: string,
  ) {}

  synthesize(request: SynthesisRequest): Observable<SynthesisResult> {
    return this.http.post<SynthesisResult>(`${this.apiBaseUrl}/synthesis`, request);
  }

  enqueue(request: SynthesisRequest, idempotencyKey: string): Observable<SynthesisJob> {
    const headers = new HttpHeaders({ 'Idempotency-Key': idempotencyKey });
    return this.http.post<SynthesisJob>(
      `${this.apiBaseUrl}/synthesis/jobs`,
      request,
      { headers },
    );
  }

  getJob(jobId: string): Observable<SynthesisJob> {
    return this.http.get<SynthesisJob>(`${this.apiBaseUrl}/synthesis/jobs/${jobId}`);
  }

  cancelJob(jobId: string): Observable<SynthesisJob> {
    return this.http.post<SynthesisJob>(
      `${this.apiBaseUrl}/synthesis/jobs/${jobId}/cancel`,
      {},
    );
  }

  listModels(): Observable<readonly ModelSummary[]> {
    return this.http.get<readonly ModelSummary[]>(`${this.apiBaseUrl}/models`);
  }
}
