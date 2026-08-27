import { HttpErrorResponse } from '@angular/common/http';
import {
  ChangeDetectionStrategy,
  Component,
  Input,
  OnDestroy,
  signal,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Subscription, switchMap, takeWhile, timer } from 'rxjs';

import { ExperimentApiService } from '../api/experiment-api.service';
import { ComparisonResultCardComponent } from './comparison-result-card.component';
import {
  ExperimentComparisonJob,
  ExperimentComparisonRequest,
  ExperimentFixture,
  ExperimentModelId,
  ExperimentModelSummary,
} from './experiment.types';

@Component({
  selector: 'ovl-live-comparison',
  standalone: true,
  imports: [FormsModule, ComparisonResultCardComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  styleUrl: './live-comparison.component.css',
  template: `
    <section class="live-lab" aria-labelledby="live-lab-title">
      <div class="section-label">Live CPU evaluation</div>
      <div class="heading-row">
        <div>
          <h3 id="live-lab-title">Test the adaptation yourself.</h3>
          <p>One text. Same speaker profile. Same CPU. Direct model comparison.</p>
        </div>
        <span class="runtime-badge"><i aria-hidden="true"></i> Self-hosted CPU</span>
      </div>

      <div class="mode-switch" role="group" aria-label="Evaluation input mode">
        <button type="button" [class.active]="mode() === 'fixture'" [attr.aria-pressed]="mode() === 'fixture'" [disabled]="isRunning()" (click)="mode.set('fixture')">
          Locked fixture
        </button>
        <button type="button" [class.active]="mode() === 'custom'" [attr.aria-pressed]="mode() === 'custom'" [disabled]="isRunning()" (click)="mode.set('custom')">
          Custom text
        </button>
      </div>

      <form (submit)="run($event)" novalidate>
        @if (mode() === 'fixture') {
          <label for="fixture-select">Locked medical sentence</label>
          <select id="fixture-select" [disabled]="isRunning()" [ngModel]="fixtureId()" (ngModelChange)="fixtureId.set($event)" name="fixtureId">
            @for (fixture of fixtures; track fixture.id) {
              <option [value]="fixture.id">{{ fixture.text }}</option>
            }
          </select>
          @if (selectedFixture(); as fixture) {
            <div class="term-preview" aria-label="Fixture target terms">
              @for (term of fixture.targetTerms; track term.canonical) {
                <span>{{ term.canonical }} · {{ term.category }}</span>
              }
            </div>
          }
        } @else {
          <label for="custom-text">Text to synthesize</label>
          <textarea id="custom-text" rows="4" maxlength="500" [disabled]="isRunning()" [ngModel]="customText()" (ngModelChange)="customText.set($event)" name="customText" aria-describedby="custom-text-help"></textarea>
          <p id="custom-text-help" class="field-help">Use a short English sentence. Processing rules remain backend-owned.</p>

          <label for="target-terms">Target terms</label>
          <input id="target-terms" type="text" [disabled]="isRunning()" [ngModel]="customTerms()" (ngModelChange)="customTerms.set($event)" name="customTerms" placeholder="amlodipine, hypertension" aria-describedby="target-terms-help" />
          <p id="target-terms-help" class="field-help">Comma-separated terms that appear exactly in the text above.</p>
        }

        <fieldset class="processing-options">
          <legend>Text processing</legend>
          <label class="toggle-row">
            <span><strong>Sanitize noisy characters</strong><small>Remove isolated symbols and repeated punctuation.</small></span>
            <input type="checkbox" [disabled]="isRunning()" [checked]="sanitizeText()" (change)="sanitizeText.set($any($event.target).checked)" />
          </label>
          <label class="toggle-row">
            <span><strong>Normalize technical text</strong><small>Convert meaningful notation into speakable English.</small></span>
            <input type="checkbox" [disabled]="isRunning()" [checked]="normalizeText()" (change)="normalizeText.set($any($event.target).checked)" />
          </label>
        </fieldset>

        <fieldset class="model-options">
          <legend>Models to compare</legend>
          <div class="model-grid">
            @for (model of models; track model.id) {
              <label [class.unavailable]="!model.available">
                <input type="checkbox" [disabled]="isRunning() || !model.available" [checked]="isSelected(model.id)" (change)="toggleModel(model.id, $any($event.target).checked)" />
                <span><strong>{{ model.name }}</strong><small>{{ model.role }} · {{ model.runtime }}</small></span>
              </label>
            }
          </div>
        </fieldset>

        @if (validationError(); as error) {
          <p class="form-error" role="alert">{{ error }}</p>
        }

        <div class="action-row">
          <button class="primary" type="submit" [disabled]="isRunning() || selectedModels().length < 2">
            {{ isRunning() ? 'Comparison running' : 'Compare models' }}
          </button>
          @if (isRunning()) {
            <button class="secondary" type="button" (click)="cancel()">Cancel after current operation</button>
          }
        </div>
      </form>

      @if (job(); as current) {
        <section class="job-panel" aria-labelledby="job-results-title" aria-live="polite">
          <div class="job-progress">
            <div><span>{{ stageLabel(current.stage) }}</span><strong>{{ current.progressPercent.toFixed(0) }}%</strong></div>
            <progress [value]="current.progressPercent" max="100">{{ current.progressPercent }}%</progress>
            @if (current.normalizedText && current.normalizedText !== current.originalText) {
              <p><strong>Text sent to models:</strong> {{ current.normalizedText }}</p>
            }
          </div>
          <h4 id="job-results-title">Live comparison</h4>
          <div class="result-grid">
            @for (result of current.results; track result.modelId) {
              <ovl-comparison-result-card [result]="result" [models]="models" />
            }
          </div>
        </section>
      }
    </section>
  `,
})
export class LiveComparisonComponent implements OnDestroy {
  private modelValues: readonly ExperimentModelSummary[] = [];
  private fixtureValues: readonly ExperimentFixture[] = [];
  private readonly subscriptions = new Subscription();
  private eventSource?: EventSource;

  readonly mode = signal<'fixture' | 'custom'>('fixture');
  readonly fixtureId = signal('');
  readonly customText = signal('The patient was prescribed amlodipine for hypertension.');
  readonly customTerms = signal('amlodipine, hypertension');
  readonly sanitizeText = signal(true);
  readonly normalizeText = signal(true);
  readonly selectedModels = signal<readonly ExperimentModelId[]>([]);
  readonly job = signal<ExperimentComparisonJob | null>(null);
  readonly validationError = signal('');

  constructor(private readonly api: ExperimentApiService) {}

  @Input({ required: true })
  set models(value: readonly ExperimentModelSummary[]) {
    this.modelValues = value;
    if (this.selectedModels().length === 0) {
      this.selectedModels.set(value.filter((model) => model.available).map((model) => model.id));
    }
  }
  get models(): readonly ExperimentModelSummary[] {
    return this.modelValues;
  }

  @Input({ required: true })
  set fixtures(value: readonly ExperimentFixture[]) {
    this.fixtureValues = value;
    if (!this.fixtureId() && value.length > 0) {
      this.fixtureId.set(value[0].id);
    }
  }
  get fixtures(): readonly ExperimentFixture[] {
    return this.fixtureValues;
  }

  selectedFixture(): ExperimentFixture | undefined {
    return this.fixtures.find((fixture) => fixture.id === this.fixtureId());
  }

  isSelected(modelId: ExperimentModelId): boolean {
    return this.selectedModels().includes(modelId);
  }

  toggleModel(modelId: ExperimentModelId, selected: boolean): void {
    const values = this.selectedModels();
    this.selectedModels.set(
      selected ? [...values, modelId] : values.filter((candidate) => candidate !== modelId),
    );
  }

  isRunning(): boolean {
    const stage = this.job()?.stage;
    return !!stage && !['completed', 'completed_with_failures', 'failed', 'cancelled'].includes(stage);
  }

  run(event: Event): void {
    event.preventDefault();
    this.validationError.set('');
    const request = this.buildRequest();
    if (!request) return;
    this.stopWatching();
    this.subscriptions.add(
      this.api.startComparison(request).subscribe({
        next: (job) => {
          this.job.set(job);
          this.watch(job.id);
        },
        error: (error: HttpErrorResponse) => this.validationError.set(this.describeError(error)),
      }),
    );
  }

  cancel(): void {
    const current = this.job();
    if (!current) return;
    this.subscriptions.add(this.api.cancelComparison(current.id).subscribe((job) => this.job.set(job)));
  }

  stageLabel(stage: ExperimentComparisonJob['stage']): string {
    return stage.replaceAll('_', ' ');
  }

  ngOnDestroy(): void {
    this.stopWatching();
    this.subscriptions.unsubscribe();
  }

  private buildRequest(): ExperimentComparisonRequest | null {
    if (this.selectedModels().length < 2) {
      this.validationError.set('Select at least two available models.');
      return null;
    }
    const base = {
      modelIds: this.selectedModels(),
      sanitizeText: this.sanitizeText(),
      normalizeText: this.normalizeText(),
    };
    if (this.mode() === 'fixture') {
      if (!this.selectedFixture()) {
        this.validationError.set('Select a locked fixture.');
        return null;
      }
      return { ...base, mode: 'fixture', fixtureId: this.fixtureId() };
    }
    const text = this.customText().trim();
    const terms = [...new Set(this.customTerms().split(',').map((term) => term.trim()).filter(Boolean))];
    if (!text || terms.length === 0) {
      this.validationError.set('Custom evaluation requires text and at least one target term.');
      return null;
    }
    const absent = terms.filter((term) => !text.toLocaleLowerCase().includes(term.toLocaleLowerCase()));
    if (absent.length > 0) {
      this.validationError.set(`Every target term must appear in the text: ${absent.join(', ')}`);
      return null;
    }
    return { ...base, mode: 'custom', text, targetTerms: terms };
  }

  private watch(jobId: string): void {
    if (typeof EventSource !== 'undefined') {
      this.eventSource = new EventSource(this.api.comparisonEventsUrl(jobId));
      this.eventSource.onmessage = (event) => {
        const job = JSON.parse(event.data) as ExperimentComparisonJob;
        this.job.set(job);
        if (!this.isRunning()) this.stopWatching();
      };
      this.eventSource.onerror = () => {
        this.eventSource?.close();
        this.eventSource = undefined;
        this.poll(jobId);
      };
      return;
    }
    this.poll(jobId);
  }

  private poll(jobId: string): void {
    this.subscriptions.add(
      timer(0, 1_000)
        .pipe(
          switchMap(() => this.api.getComparison(jobId)),
          takeWhile(
            (job) => !['completed', 'completed_with_failures', 'failed', 'cancelled'].includes(job.stage),
            true,
          ),
        )
        .subscribe({
          next: (job) => this.job.set(job),
          error: (error: HttpErrorResponse) => this.validationError.set(this.describeError(error)),
        }),
    );
  }

  private stopWatching(): void {
    this.eventSource?.close();
    this.eventSource = undefined;
  }

  private describeError(error: HttpErrorResponse): string {
    if (error.status === 0) return 'Backend unavailable. Keep your text and restart FastAPI.';
    if (error.status === 429) return 'The CPU queue is full. Wait for the active comparison.';
    const detail = error.error?.detail;
    return typeof detail === 'string' ? detail : 'The comparison could not be started.';
  }
}
