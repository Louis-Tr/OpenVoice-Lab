import { HttpErrorResponse } from '@angular/common/http';
import {
  ChangeDetectionStrategy,
  Component,
  computed,
  OnDestroy,
  OnInit,
  signal,
} from '@angular/core';
import { Subscription, switchMap, takeWhile, tap, timer } from 'rxjs';

import {
  BenchmarkApiService,
  BenchmarkConfig,
  BenchmarkJobStatus,
} from '../api/benchmark-api.service';
import {
  BenchmarkResultsComponent,
  BenchmarkResultsState,
} from './benchmark-results.component';

@Component({
  selector: 'ovl-benchmark-page',
  standalone: true,
  imports: [BenchmarkResultsComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  styleUrl: './benchmark-page.component.css',
  template: `
    <section class="benchmark-lab" aria-labelledby="benchmark-title">
      <header class="hero">
        <p class="eyebrow"><i aria-hidden="true"></i> Reproducible evaluation</p>
        <h2 id="benchmark-title">Benchmark <span>deployment choices.</span></h2>
        <p class="intro">
          Run one fixed workload against every local model configuration. Compare only the numbers that change a deployment decision.
        </p>
      </header>

      <section class="run-card" aria-labelledby="run-heading">
        <div class="card-heading">
          <span>01 / WORKLOAD</span>
          @if (config()) {
            <span>Corpus {{ config()?.corpusVersion }}</span>
          }
        </div>
        <div class="run-heading-row">
          <div>
            <h3 id="run-heading">Fixed benchmark corpus</h3>
            <p>Every variant receives the same text, voice, and case order.</p>
          </div>
          <button
            type="button"
            [disabled]="configState() !== 'ready' || isRunning()"
            (click)="runBenchmark()"
          >
            @if (isRunning()) {
              Benchmark running
            } @else {
              Run benchmark
            }
          </button>
        </div>

        <dl class="workload-facts">
          <div>
            <dt>Test cases</dt>
            <dd>{{ config()?.testCaseCount ?? '—' }}</dd>
          </div>
          <div>
            <dt>Models</dt>
            <dd>{{ config()?.modelCount ?? '—' }}</dd>
          </div>
          <div>
            <dt>Evaluations</dt>
            <dd>{{ config()?.totalEvaluations ?? '—' }}</dd>
          </div>
          <div>
            <dt>Voice</dt>
            <dd class="voice">{{ config()?.defaultVoiceId ?? '—' }}</dd>
          </div>
        </dl>

        @if (configState() === 'loading') {
          <p class="inline-status" aria-live="polite">Loading benchmark contract…</p>
        }

        @if (isRunning() && job(); as activeJob) {
          <div class="progress-panel" aria-live="polite">
            <div class="progress-copy">
              <span>{{ activeJob.status === 'pending' ? 'Preparing workers' : 'Running fixed workload' }}</span>
              <strong>
                {{ activeJob.completedEvaluations }} / {{ activeJob.totalEvaluations }} evaluations
              </strong>
            </div>
            <progress
              [value]="activeJob.completedEvaluations"
              [max]="activeJob.totalEvaluations"
              [attr.aria-label]="activeJob.progressPercent + '% complete'"
            ></progress>
            <p>Models run in isolated processes. This can take several minutes on CPU.</p>
          </div>
        }

        @if (job()?.status === 'completed') {
          <div class="completion-state" role="status">
            Benchmark complete. Comparative results are ready below.
          </div>
        }

        @if (requestError()) {
          <div class="error-state" role="alert">
            <div>
              <strong>Benchmark unavailable.</strong>
              <p>{{ requestError() }}</p>
            </div>
            <button type="button" (click)="recover()">
              {{ configState() === 'error' ? 'Retry connection' : 'Try again' }}
            </button>
          </div>
        }
      </section>

      <ovl-benchmark-results
        [aggregates]="job()?.result?.aggregates ?? []"
        [state]="resultsState()"
      />

      <footer class="proof-strip" aria-label="Benchmark guarantees">
        <span>Fixed corpus</span>
        <i aria-hidden="true"></i>
        <span>Isolated models</span>
        <i aria-hidden="true"></i>
        <span>Recorded failures</span>
      </footer>
    </section>
  `,
})
export class BenchmarkPageComponent implements OnInit, OnDestroy {
  readonly config = signal<BenchmarkConfig | null>(null);
  readonly configState = signal<'loading' | 'ready' | 'error'>('loading');
  readonly job = signal<BenchmarkJobStatus | null>(null);
  readonly requestError = signal('');

  readonly isRunning = computed(() => {
    const status = this.job()?.status;
    return status === 'pending' || status === 'running';
  });

  readonly resultsState = computed<BenchmarkResultsState>(() => {
    const status = this.job()?.status;
    if (status === 'pending' || status === 'running') {
      return 'loading';
    }
    if (status === 'completed') {
      return 'ready';
    }
    if (status === 'failed' || this.requestError()) {
      return 'error';
    }
    return 'empty';
  });

  private readonly subscriptions = new Subscription();
  private pollSubscription?: Subscription;

  constructor(private readonly benchmarkApi: BenchmarkApiService) {}

  ngOnInit(): void {
    this.loadConfig();
  }

  ngOnDestroy(): void {
    this.pollSubscription?.unsubscribe();
    this.subscriptions.unsubscribe();
  }

  loadConfig(): void {
    this.configState.set('loading');
    this.requestError.set('');
    this.subscriptions.add(
      this.benchmarkApi.getConfig().subscribe({
        next: (config) => {
          this.config.set(config);
          this.configState.set('ready');
          this.loadLatestBenchmark();
        },
        error: (error: HttpErrorResponse) => {
          this.configState.set('error');
          this.requestError.set(this.describeError(error));
        },
      }),
    );
  }

  runBenchmark(): void {
    const config = this.config();
    if (!config || this.isRunning()) {
      return;
    }

    this.pollSubscription?.unsubscribe();
    this.job.set(null);
    this.requestError.set('');
    this.subscriptions.add(
      this.benchmarkApi.runBenchmark({
        modelIds: config.modelIds,
        voiceId: config.defaultVoiceId,
      })
      .subscribe({
        next: (job) => {
          this.job.set(job);
          this.pollBenchmark(job.benchmarkId);
        },
        error: (error: HttpErrorResponse) => {
          this.requestError.set(this.describeError(error));
        },
      }),
    );
  }

  recover(): void {
    if (this.configState() === 'error') {
      this.loadConfig();
      return;
    }
    this.runBenchmark();
  }

  private loadLatestBenchmark(): void {
    this.subscriptions.add(
      this.benchmarkApi.getLatestBenchmark().subscribe({
        next: (job) => {
          this.job.set(job);
          if (job.status === 'pending' || job.status === 'running') {
            this.pollBenchmark(job.benchmarkId);
          } else if (job.status === 'failed') {
            this.requestError.set(
              job.error || 'The benchmark worker stopped before producing results.',
            );
          }
        },
        error: (error: HttpErrorResponse) => {
          if (error.status !== 404) {
            this.requestError.set(this.describeError(error));
          }
        },
      }),
    );
  }

  private pollBenchmark(identifier: string): void {
    this.pollSubscription?.unsubscribe();
    this.pollSubscription = timer(0, 2_000)
      .pipe(
        switchMap(() => this.benchmarkApi.getBenchmark(identifier)),
        tap((job) => {
          this.job.set(job);
          if (job.status === 'failed') {
            this.requestError.set(
              job.error || 'The benchmark worker stopped before producing results.',
            );
          }
        }),
        takeWhile(
          (job) => job.status === 'pending' || job.status === 'running',
          true,
        ),
      )
      .subscribe({
        error: (error: HttpErrorResponse) => {
          this.requestError.set(this.describeError(error));
        },
      });
    this.subscriptions.add(this.pollSubscription);
  }

  private describeError(error: HttpErrorResponse): string {
    if (error.status === 0) {
      return 'Backend unavailable. Start FastAPI, then retry the connection.';
    }
    if (error.status === 404) {
      return 'The benchmark job was lost after a backend restart. Run it again.';
    }
    return 'The evaluation pipeline failed. Check the backend logs, then try again.';
  }
}
