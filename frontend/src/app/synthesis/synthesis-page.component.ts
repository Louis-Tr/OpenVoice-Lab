import { HttpErrorResponse } from '@angular/common/http';
import {
  ChangeDetectionStrategy,
  Component,
  computed,
  OnDestroy,
  OnInit,
  signal,
  ViewChild,
} from '@angular/core';
import { exhaustMap, Subscription, takeWhile, timer } from 'rxjs';

import { AudioPlayerComponent } from '../audio-player/audio-player.component';
import { SynthesisApiService } from '../api/synthesis-api.service';
import { InferenceMetricsComponent } from '../metrics/inference-metrics.component';
import { ModelSelectorComponent } from '../model-selector/model-selector.component';
import { SynthesisFormComponent } from './synthesis-form.component';
import { SynthesisSession, SynthesisSessionStore } from './synthesis-session.store';
import {
  isTerminalJob,
  ModelSelection,
  ModelSummary,
  SynthesisJob,
  SynthesisResult,
} from './synthesis.types';

@Component({
  selector: 'ovl-synthesis-page',
  standalone: true,
  imports: [
    AudioPlayerComponent,
    InferenceMetricsComponent,
    ModelSelectorComponent,
    SynthesisFormComponent,
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
  styleUrl: './synthesis-page.component.css',
  template: `
    <section class="lab" aria-labelledby="synthesis-title">
      <header class="hero">
        <p class="eyebrow"><i aria-hidden="true"></i> Self-hosted synthesis</p>
        <h2 id="synthesis-title">Turn text into <span>local speech.</span></h2>
        <p class="intro">
          Choose a deployed voice. Generate one clean WAV. Play it without sending text to an external inference API.
        </p>
      </header>

      <div class="workspace">
        <section class="request-card" aria-labelledby="request-heading">
          <p class="card-index">01 / INPUT</p>
          <h3 id="request-heading">Synthesis request</h3>

          <ovl-synthesis-form
            [text]="text()"
            [error]="textError() || inputLimitError()"
            [maxLength]="inputLimit()"
            [disabled]="isSubmitting()"
            [submitting]="isSubmitting()"
            [canSubmit]="modelState() === 'ready' && !inputLimitError()"
            [sanitizeText]="sanitizeText()"
            [normalizeText]="normalizeText()"
            (textChange)="setText($event)"
            (sanitizeTextChange)="sanitizeText.set($event)"
            (normalizeTextChange)="normalizeText.set($event)"
            (generate)="submit()"
          >
            <ovl-model-selector
              [models]="models()"
              [selectedModelId]="selectedModelId()"
              [selectedVoiceId]="selectedVoiceId()"
              [state]="modelState()"
              [disabled]="isSubmitting()"
              (modelSelectionChange)="setModelSelection($event)"
              (voiceIdChange)="selectedVoiceId.set($event)"
            />
          </ovl-synthesis-form>

          @if (generationStatus()) {
            <p class="generation-status" role="status">{{ generationStatus() }}</p>
          }
          @if (storageWarning()) {
            <p class="generation-status" role="alert">{{ storageWarning() }}</p>
          }
          @if (requestError()) {
            <div class="request-error" role="alert">
              <div>
                <strong>{{ needsReconnect() ? 'Connection interrupted.' : 'Request stopped.' }}</strong>
                <p>{{ requestError() }}</p>
              </div>
              @if (needsReconnect()) {
                <button type="button" (click)="reconnect()">Check generation</button>
              } @else if (modelState() === 'error') {
                <button type="button" (click)="loadModels()">Retry connection</button>
              }
            </div>
          }
        </section>

        <div class="output-stack">
          <ovl-audio-player [result]="result()" />
          @if (processedTextPreview(); as processedText) {
            <section class="processed-preview" aria-labelledby="processed-text-heading">
              <p id="processed-text-heading">Text sent to model</p>
              <blockquote>{{ processedText }}</blockquote>
            </section>
          }
          <ovl-inference-metrics [metrics]="result()?.metrics ?? null" />
        </div>
      </div>

      <footer class="proof-strip" aria-label="Workflow guarantees">
        <span>Browser contract</span>
        <i aria-hidden="true"></i>
        <span>Local model</span>
        <i aria-hidden="true"></i>
        <span>Measured inference</span>
      </footer>
    </section>
  `,
})
export class SynthesisPageComponent implements OnInit, OnDestroy {
  readonly models = signal<readonly ModelSummary[]>([]);
  readonly modelState = signal<'loading' | 'ready' | 'error'>('loading');
  readonly selectedModelId = signal('');
  readonly selectedVoiceId = signal('');
  readonly text = signal('');
  readonly textError = signal('');
  readonly requestError = signal('');
  readonly isSubmitting = signal(false);
  readonly sanitizeText = signal(true);
  readonly normalizeText = signal(true);
  readonly result = signal<SynthesisResult | null>(null);
  readonly generationStatus = signal('');
  readonly needsReconnect = signal(false);
  readonly storageWarning = signal('');

  readonly selectedModel = computed(() =>
    this.models().find((model) => model.id === this.selectedModelId()),
  );

  readonly inputLimit = computed(() => this.selectedModel()?.maxInputCharacters ?? 5000);
  readonly inputLimitError = computed(() =>
    this.text().length > this.inputLimit()
      ? `${this.selectedModel()?.name ?? 'This model'} accepts up to ${this.inputLimit()} characters. Shorten the text or choose another model.`
      : '',
  );

  readonly processedTextPreview = computed(() => this.result()?.normalizedText ?? null);

  @ViewChild(SynthesisFormComponent) private synthesisForm?: SynthesisFormComponent;

  private readonly subscriptions = new Subscription();
  private observation = new Subscription();
  private session: SynthesisSession | null = null;

  constructor(
    private readonly synthesisApi: SynthesisApiService,
    private readonly sessionStore: SynthesisSessionStore,
  ) {}

  ngOnInit(): void {
    this.session = this.sessionStore.load();
    if (this.session) {
      const request = this.session.request;
      this.text.set(request.text);
      this.selectedModelId.set(request.modelId);
      this.selectedVoiceId.set(request.voiceId);
      this.sanitizeText.set(request.sanitizeText);
      this.normalizeText.set(request.normalizeText);
    }
    this.loadModels();
    // Recovery must work even if the current model catalog cannot be loaded.
    if (this.session) this.reconnect();
  }

  ngOnDestroy(): void {
    this.subscriptions.unsubscribe();
    this.observation.unsubscribe();
  }

  loadModels(): void {
    this.modelState.set('loading');
    this.requestError.set('');

    this.subscriptions.add(
      this.synthesisApi.listModels().subscribe({
        next: (models) => {
          const availableModels = models.filter((model) => model.available);
          this.models.set(models);

          if (availableModels.length === 0) {
            this.modelState.set('error');
            this.requestError.set(
              'No model artifacts are ready. Provision the backend model files, then retry.',
            );
            return;
          }

          if (!this.selectedModelId()) {
            const firstModel = availableModels[0];
            this.selectedModelId.set(firstModel.id);
            this.selectedVoiceId.set(firstModel.voices[0] ?? '');
          }
          this.modelState.set('ready');
        },
        error: (error: HttpErrorResponse) => {
          this.modelState.set('error');
          this.requestError.set(this.describeModelError(error));
        },
      }),
    );
  }

  setText(text: string): void {
    this.text.set(text);
    if (this.textError()) {
      this.textError.set('');
    }
  }

  setModelSelection(selection: ModelSelection): void {
    const model = this.models().find((item) => item.id === selection.modelId);
    if (!model?.available) {
      return;
    }
    this.selectedModelId.set(selection.modelId);
    if (!model.voices.includes(this.selectedVoiceId())) {
      this.selectedVoiceId.set(model.voices[0] ?? '');
    }
  }

  submit(): void {
    if (this.isSubmitting()) return;
    if (this.inputLimitError()) {
      this.synthesisForm?.focusText();
      return;
    }
    const text = this.text().trim();
    if (!text) {
      this.textError.set('Enter text before generating speech.');
      this.synthesisForm?.focusText();
      return;
    }

    const model = this.selectedModel();
    if (!model?.available || !this.selectedVoiceId()) {
      this.requestError.set('Choose an available model and voice, then try again.');
      return;
    }

    this.text.set(text);
    this.textError.set('');
    this.requestError.set('');
    const session: SynthesisSession = {
      version: 1,
      key: crypto.randomUUID(),
      createdAt: Date.now(),
      jobId: null,
      stoppedReason: null,
      request: {
        text,
        modelId: model.id,
        voiceId: this.selectedVoiceId(),
        sanitizeText: this.sanitizeText(),
        normalizeText: this.normalizeText(),
      },
    };
    // Persist BEFORE POST so a lost response can be retried using this same key.
    if (!this.sessionStore.save(session)) {
      this.requestError.set('Could not save this request in your browser. Enable site storage and try again.');
      return;
    }
    this.storageWarning.set('');
    this.result.set(null);
    this.session = session;
    this.reconnect();
  }

  reconnect(): void {
    const session = this.session;
    if (!session) return;
    this.observation.unsubscribe();
    this.observation = new Subscription();
    this.needsReconnect.set(false);
    if (session.stoppedReason) {
      this.isSubmitting.set(false);
      this.requestError.set(session.stoppedReason);
      return;
    }
    if (!session.jobId && Date.now() - session.createdAt >= 86_400_000) {
      this.stopSession('This saved submission is too old to retry automatically. Generate again to start a new request.');
      return;
    }
    this.isSubmitting.set(true);
    this.requestError.set('');
    this.generationStatus.set(session.jobId ? 'Restoring your generation…' : 'Submitting your request…');
    if (session.jobId) {
      this.watchJob(session.jobId);
    } else {
      this.observation.add(this.synthesisApi.startJob(session.request, session.key).subscribe({
        next: (job) => {
          this.applyJob(job);
          if (!isTerminalJob(job)) this.watchJob(job.jobId, 2000);
        },
        error: (error: HttpErrorResponse) => {
          const backendRejection = [500, 503].includes(error.status)
            && typeof error.error?.detail === 'string';
          if ([400, 401, 403, 404, 409, 422].includes(error.status) || backendRejection) {
            this.stopSession(this.describeSynthesisError(error));
          } else {
            this.connectionInterrupted();
          }
        },
      }));
    }
  }

  private watchJob(jobId: string, delay = 0): void {
    this.observation.add(timer(delay, 2000).pipe(
      // Slow status requests must finish rather than being cancelled every tick.
      exhaustMap(() => this.synthesisApi.getJob(jobId)),
      takeWhile((job) => !isTerminalJob(job), true),
    ).subscribe({
      next: (job) => this.applyJob(job),
      error: (error: HttpErrorResponse) => {
        if (error.status === 404 || error.status === 410) {
          this.stopSession('This generation is no longer available. It may have expired or the backend restarted. Your input is restored; generate again to start a new request.');
        } else {
          this.connectionInterrupted();
        }
      },
    }));
  }

  private applyJob(job: SynthesisJob): void {
    if (!this.session) return;
    this.persist({ ...this.session, jobId: job.jobId });
    this.needsReconnect.set(false);
    this.requestError.set('');
    this.isSubmitting.set(!isTerminalJob(job));
    const labels: Record<SynthesisJob['status'], string> = {
      pending: 'Checking available resources…', loading: 'Loading the model…',
      generating: 'Generating your audio…', saving: 'Saving your audio…',
      completed: 'Your audio is ready.', failed: 'Generation failed.', rejected: 'Request not accepted.',
    };
    this.generationStatus.set(labels[job.status]);
    if (job.status === 'completed') this.result.set(job.result);
    if (job.status === 'failed' || job.status === 'rejected') {
      this.requestError.set(job.error?.detail ?? 'Generation failed. Try a new request.');
    }
  }

  private connectionInterrupted(): void {
    // Unknown outcome: keep the key and disable new submissions until checked.
    this.needsReconnect.set(true);
    this.generationStatus.set('Generation status is temporarily unavailable.');
    this.requestError.set('The backend may still be generating your audio. Check generation to reconnect to this same request.');
  }

  private stopSession(reason: string): void {
    if (this.session) this.persist({ ...this.session, stoppedReason: reason });
    this.isSubmitting.set(false);
    this.needsReconnect.set(false);
    this.generationStatus.set('');
    this.requestError.set(reason);
  }

  private persist(session: SynthesisSession): void {
    this.session = session;
    this.storageWarning.set(this.sessionStore.save(session) ? ''
      : 'Browser storage could not be updated. Keep this page open until generation finishes.');
  }

  private describeModelError(error: HttpErrorResponse): string {
    if (error.status === 0) {
      return 'Backend unavailable. Start FastAPI on port 8000, then retry the connection.';
    }
    return 'Models could not be loaded. Check the backend logs, then retry the connection.';
  }

  private describeSynthesisError(error: HttpErrorResponse): string {
    if (typeof error.error?.detail === 'string') return error.error.detail;
    if (error.status === 0) {
      return 'Backend unavailable. Keep this text, restart FastAPI, and submit again.';
    }
    if (error.status === 422) {
      return 'The backend rejected this text or selection. Review the fields and submit again.';
    }
    if (error.status === 404) {
      return 'That model is no longer available. Reload the page to refresh model choices.';
    }
    if (error.status === 503) {
      return 'The local model is not ready. Provision its artifacts, then submit again.';
    }
    return 'Inference failed before audio was created. Check the backend logs and submit again.';
  }
}
