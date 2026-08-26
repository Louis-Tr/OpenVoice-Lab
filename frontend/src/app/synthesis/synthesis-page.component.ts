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
import { finalize, Subscription } from 'rxjs';

import { AudioPlayerComponent } from '../audio-player/audio-player.component';
import { SynthesisApiService } from '../api/synthesis-api.service';
import { InferenceMetricsComponent } from '../metrics/inference-metrics.component';
import { ModelSelectorComponent } from '../model-selector/model-selector.component';
import { SynthesisFormComponent } from './synthesis-form.component';
import {
  ModelSelection,
  ModelSummary,
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
            [error]="textError()"
            [disabled]="isSubmitting()"
            [submitting]="isSubmitting()"
            [canSubmit]="modelState() === 'ready'"
            [sanitizeText]="sanitizeText()"
            (textChange)="setText($event)"
            (sanitizeTextChange)="sanitizeText.set($event)"
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

          @if (requestError()) {
            <div class="request-error" role="alert">
              <div>
                <strong>Request stopped.</strong>
                <p>{{ requestError() }}</p>
              </div>
              @if (modelState() === 'error') {
                <button type="button" (click)="loadModels()">Retry connection</button>
              }
            </div>
          }
        </section>

        <div class="output-stack">
          <ovl-audio-player [result]="result()" />
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
  readonly result = signal<SynthesisResult | null>(null);

  readonly selectedModel = computed(() =>
    this.models().find((model) => model.id === this.selectedModelId()),
  );

  @ViewChild(SynthesisFormComponent) private synthesisForm?: SynthesisFormComponent;

  private readonly subscriptions = new Subscription();

  constructor(private readonly synthesisApi: SynthesisApiService) {}

  ngOnInit(): void {
    this.loadModels();
  }

  ngOnDestroy(): void {
    this.subscriptions.unsubscribe();
  }

  loadModels(): void {
    this.modelState.set('loading');
    this.requestError.set('');

    this.subscriptions.add(
      this.synthesisApi.listModels().subscribe({
        next: (models) => {
          const availableModels = models.filter((model) => model.available);
          this.models.set(availableModels);

          if (availableModels.length === 0) {
            this.modelState.set('error');
            this.requestError.set(
              'No model artifacts are ready. Provision the backend model files, then retry.',
            );
            return;
          }

          const firstModel = availableModels[0];
          this.selectedModelId.set(firstModel.id);
          this.selectedVoiceId.set(firstModel.voices[0] ?? '');
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
    this.selectedModelId.set(selection.modelId);

    const model = this.models().find((item) => item.id === selection.modelId);
    if (model && !model.voices.includes(this.selectedVoiceId())) {
      this.selectedVoiceId.set(model.voices[0] ?? '');
    }
  }

  submit(): void {
    const text = this.text().trim();
    if (!text) {
      this.textError.set('Enter text before generating speech.');
      this.synthesisForm?.focusText();
      return;
    }

    const model = this.selectedModel();
    if (!model || !this.selectedVoiceId()) {
      this.requestError.set('Choose an available model and voice, then try again.');
      return;
    }

    this.text.set(text);
    this.textError.set('');
    this.requestError.set('');
    this.result.set(null);
    this.isSubmitting.set(true);

    this.subscriptions.add(
      this.synthesisApi
        .synthesize({
          text,
          modelId: model.id,
          voiceId: this.selectedVoiceId(),
          sanitizeText: this.sanitizeText(),
        })
        .pipe(finalize(() => this.isSubmitting.set(false)))
        .subscribe({
          next: (result) => this.result.set(result),
          error: (error: HttpErrorResponse) => {
            this.requestError.set(this.describeSynthesisError(error));
          },
        }),
    );
  }

  private describeModelError(error: HttpErrorResponse): string {
    if (error.status === 0) {
      return 'Backend unavailable. Start FastAPI on port 8000, then retry the connection.';
    }
    return 'Models could not be loaded. Check the backend logs, then retry the connection.';
  }

  private describeSynthesisError(error: HttpErrorResponse): string {
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
