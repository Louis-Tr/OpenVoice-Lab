import {
  ChangeDetectionStrategy,
  Component,
  EventEmitter,
  Input,
  Output,
} from '@angular/core';

import {
  ModelSelection,
  ModelSummary,
  ModelVariant,
} from '../synthesis/synthesis.types';

interface ModelChoice extends ModelSelection {
  readonly key: string;
  readonly label: string;
}

@Component({
  selector: 'ovl-model-selector',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  styleUrl: './model-selector.component.css',
  template: `
    <fieldset [disabled]="disabled || state !== 'ready'">
      <legend>Voice configuration</legend>
      <div class="field-grid">
        <label>
          <span>Voice</span>
          <select
            aria-describedby="model-status"
            [value]="selectedVoiceId"
            (change)="onVoiceChange($event)"
          >
            @for (voice of voices; track voice) {
              <option [value]="voice">{{ voice }}</option>
            }
          </select>
        </label>

        <label>
          <span>Model</span>
          <select
            aria-describedby="model-status"
            [value]="selectedChoice"
            (change)="onModelChange($event)"
          >
            @for (choice of choices; track choice.key) {
              <option [value]="choice.key">{{ choice.label }}</option>
            }
          </select>
        </label>
      </div>

      <p id="model-status" class="model-status" aria-live="polite">
        @if (state === 'loading') {
          Loading locally available models…
        } @else if (state === 'error') {
          Model discovery is unavailable.
        } @else if (selectedModel) {
          {{ selectedModel.modelVersion }} · {{ selectedModel.hosting }}
        }
      </p>
    </fieldset>
  `,
})
export class ModelSelectorComponent {
  @Input() models: readonly ModelSummary[] = [];
  @Input() selectedModelId = '';
  @Input() selectedVariant: ModelVariant = 'fp32';
  @Input() selectedVoiceId = '';
  @Input() state: 'loading' | 'ready' | 'error' = 'loading';
  @Input() disabled = false;

  @Output() readonly modelSelectionChange = new EventEmitter<ModelSelection>();
  @Output() readonly voiceIdChange = new EventEmitter<string>();

  get selectedModel(): ModelSummary | undefined {
    return this.models.find((model) => model.id === this.selectedModelId);
  }

  get voices(): readonly string[] {
    return this.selectedModel?.voices ?? [];
  }

  get choices(): readonly ModelChoice[] {
    return this.models.flatMap((model) =>
      model.variants.map((variant) => ({
        key: `${model.id}:${variant}`,
        label: `${model.displayName} ${variant.toUpperCase()}`,
        modelId: model.id,
        variant,
      })),
    );
  }

  get selectedChoice(): string {
    return `${this.selectedModelId}:${this.selectedVariant}`;
  }

  onModelChange(event: Event): void {
    const key = (event.target as HTMLSelectElement).value;
    const choice = this.choices.find((item) => item.key === key);
    if (choice) {
      this.modelSelectionChange.emit({
        modelId: choice.modelId,
        variant: choice.variant,
      });
    }
  }

  onVoiceChange(event: Event): void {
    this.voiceIdChange.emit((event.target as HTMLSelectElement).value);
  }
}
