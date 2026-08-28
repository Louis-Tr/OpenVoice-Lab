import {
  ChangeDetectionStrategy,
  Component,
  EventEmitter,
  Input,
  Output,
} from '@angular/core';

import { ModelSelection, ModelSummary } from '../synthesis/synthesis.types';

interface ModelChoice extends ModelSelection {
  readonly label: string;
  readonly precision: string;
  readonly runtime: string;
  readonly available: boolean;
  readonly unavailableReason: string | null;
}

@Component({
  selector: 'ovl-model-selector',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  styleUrl: './model-selector.component.css',
  template: `
    <fieldset [disabled]="disabled || state !== 'ready'">
      <legend>Voice configuration</legend>
      <div class="model-heading">
        <span>Model</span>
        <small>{{ readyCount }} ready / {{ choices.length }} configurations</small>
      </div>
      <div class="model-grid" role="radiogroup" aria-label="Synthesis model">
        @for (choice of choices; track choice.modelId) {
          <button
            type="button"
            role="radio"
            class="model-choice"
            [class.selected]="choice.modelId === selectedModelId"
            [class.unavailable]="!choice.available"
            [attr.aria-checked]="choice.modelId === selectedModelId"
            [attr.aria-disabled]="!choice.available"
            [attr.title]="choice.unavailableReason"
            (click)="selectChoice(choice)"
          >
            <span>{{ choice.label }}</span>
            <small>{{ choice.runtime }}</small>
            <em>{{ choice.available ? 'Ready' : 'Setup required' }}</em>
          </button>
        }
      </div>

      <p id="model-status" class="model-status" aria-live="polite">
        @if (state === 'loading') {
          Loading locally available models…
        } @else if (state === 'error') {
          Model discovery is unavailable.
        } @else if (catalogNotice) {
          {{ catalogNotice }}
        } @else if (selectedModel) {
          {{ selectedModel.description }} {{ selectedModel.precision }} · {{ selectedModel.runtime }} · {{ selectedModel.hosting }}
        }
      </p>

      <label class="voice-field">
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
    </fieldset>
  `,
})
export class ModelSelectorComponent {
  @Input() models: readonly ModelSummary[] = [];
  @Input() selectedModelId = '';
  @Input() selectedVoiceId = '';
  @Input() state: 'loading' | 'ready' | 'error' = 'loading';
  @Input() disabled = false;

  @Output() readonly modelSelectionChange = new EventEmitter<ModelSelection>();
  @Output() readonly voiceIdChange = new EventEmitter<string>();

  catalogNotice = '';

  get selectedModel(): ModelSummary | undefined {
    return this.models.find((model) => model.id === this.selectedModelId);
  }

  get voices(): readonly string[] {
    return this.selectedModel?.voices ?? [];
  }

  get choices(): readonly ModelChoice[] {
    return this.models.map((model) => ({
      label: `${model.name} ${model.precision}`,
      modelId: model.id,
      precision: model.precision,
      runtime: model.runtime,
      available: model.available,
      unavailableReason: model.unavailableReason,
    }));
  }

  get readyCount(): number {
    return this.choices.filter((choice) => choice.available).length;
  }

  selectChoice(choice: ModelChoice): void {
    if (!choice.available) {
      this.catalogNotice = choice.unavailableReason ?? `${choice.label} is not ready.`;
      return;
    }
    this.catalogNotice = '';
    this.modelSelectionChange.emit({ modelId: choice.modelId });
  }

  onModelChange(event: Event): void {
    const modelId = (event.target as HTMLSelectElement).value;
    const choice = this.choices.find((item) => item.modelId === modelId);
    if (choice) this.selectChoice(choice);
  }

  onVoiceChange(event: Event): void {
    this.voiceIdChange.emit((event.target as HTMLSelectElement).value);
  }
}
