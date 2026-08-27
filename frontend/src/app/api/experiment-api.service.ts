import { HttpClient, HttpParams } from '@angular/common/http';
import { Inject, Injectable } from '@angular/core';
import { Observable } from 'rxjs';

import { API_BASE_URL } from '../core/api-base-url.token';
import {
  ExperimentComparisonJob,
  ExperimentComparisonRequest,
  ExperimentFixturePage,
  ExperimentModelSummary,
  ExperimentReport,
} from '../experiment/experiment.types';

@Injectable({ providedIn: 'root' })
export class ExperimentApiService {
  constructor(
    private readonly http: HttpClient,
    @Inject(API_BASE_URL) private readonly apiBaseUrl: string,
  ) {}

  getReport(): Observable<ExperimentReport> {
    return this.http.get<ExperimentReport>(`${this.apiBaseUrl}/experiments/stage11/report`);
  }

  getModels(): Observable<readonly ExperimentModelSummary[]> {
    return this.http.get<readonly ExperimentModelSummary[]>(
      `${this.apiBaseUrl}/experiments/stage11/models`,
    );
  }

  getFixtures(query = '', limit = 30): Observable<ExperimentFixturePage> {
    let params = new HttpParams().set('limit', limit);
    if (query.trim()) {
      params = params.set('query', query.trim());
    }
    return this.http.get<ExperimentFixturePage>(
      `${this.apiBaseUrl}/experiments/stage11/fixtures`,
      { params },
    );
  }

  startComparison(request: ExperimentComparisonRequest): Observable<ExperimentComparisonJob> {
    return this.http.post<ExperimentComparisonJob>(
      `${this.apiBaseUrl}/experiments/stage11/comparisons`,
      request,
    );
  }

  getComparison(jobId: string): Observable<ExperimentComparisonJob> {
    return this.http.get<ExperimentComparisonJob>(
      `${this.apiBaseUrl}/experiments/stage11/comparisons/${encodeURIComponent(jobId)}`,
    );
  }

  cancelComparison(jobId: string): Observable<ExperimentComparisonJob> {
    return this.http.delete<ExperimentComparisonJob>(
      `${this.apiBaseUrl}/experiments/stage11/comparisons/${encodeURIComponent(jobId)}`,
    );
  }

  comparisonEventsUrl(jobId: string): string {
    return `${this.apiBaseUrl}/experiments/stage11/comparisons/${encodeURIComponent(jobId)}/events`;
  }
}
