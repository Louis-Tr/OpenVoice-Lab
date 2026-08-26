import { HttpClient } from '@angular/common/http';
import { Inject, Injectable } from '@angular/core';
import { Observable } from 'rxjs';

import { API_BASE_URL } from '../core/api-base-url.token';
import {
  ModelSummary,
  SynthesisRequest,
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

  listModels(): Observable<readonly ModelSummary[]> {
    return this.http.get<readonly ModelSummary[]>(`${this.apiBaseUrl}/models`);
  }
}

