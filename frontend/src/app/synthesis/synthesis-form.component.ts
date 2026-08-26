import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  EventEmitter,
  Input,
  Output,
  ViewChild,
} from '@angular/core';

@Component({
  selector: 'ovl-synthesis-form',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  styleUrl: './synthesis-form.component.css',
  template: `
    <form (submit)="onSubmit($event)" novalidate>
      <label for="synthesis-text">
        Text <span aria-hidden="true">*</span>
      </label>
      <textarea
        #textArea
        id="synthesis-text"
        name="text"
        rows="6"
        maxlength="5000"
        autocomplete="off"
        placeholder="OpenVoice Lab is running locally."
        [value]="text"
        [disabled]="disabled"
        [attr.aria-invalid]="error ? true : null"
        [attr.aria-describedby]="error ? 'synthesis-text-error synthesis-text-help' : 'synthesis-text-help'"
        (input)="onTextInput($event)"
      ></textarea>
      <div class="text-meta">
        <p id="synthesis-text-help">Enter the exact words to generate.</p>
        <span>{{ text.length }} / 5000</span>
      </div>
      @if (error) {
        <p id="synthesis-text-error" class="field-error" role="alert">{{ error }}</p>
      }

      <fieldset class="text-cleanup">
        <legend>Text cleanup</legend>
        <div class="cleanup-options">
          <label class="cleanup-option" for="sanitize-text">
            <input
              class="switch-input"
              id="sanitize-text"
              name="sanitizeText"
              type="checkbox"
              [checked]="sanitizeText"
              [disabled]="disabled"
              (change)="onSanitizeTextChange($event)"
            />
            <span class="cleanup-copy">
              <strong>Sanitize noisy characters</strong>
              <small>Remove leftover noise before local inference.</small>
            </span>
            <span class="switch-meta" aria-hidden="true">
              <span class="switch-status"></span>
              <span class="switch-track"><i></i></span>
            </span>
          </label>

          <label class="cleanup-option" for="normalize-text">
            <input
              class="switch-input"
              id="normalize-text"
              name="normalizeText"
              type="checkbox"
              [checked]="normalizeText"
              [disabled]="disabled"
              (change)="onNormalizeTextChange($event)"
            />
            <span class="cleanup-copy">
              <strong>Normalize technical text</strong>
              <small>Convert supported notation into speakable English.</small>
            </span>
            <span class="switch-meta" aria-hidden="true">
              <span class="switch-status"></span>
              <span class="switch-track"><i></i></span>
            </span>
          </label>
        </div>
      </fieldset>

      <ng-content />

      <button type="submit" [disabled]="disabled || !canSubmit">
        @if (submitting) {
          <span class="spinner" aria-hidden="true"></span>
          Generating audio…
        } @else {
          <svg aria-hidden="true" viewBox="0 0 24 24" width="20" height="20">
            <path d="M5 3l14 9-14 9V3z" fill="currentColor" />
          </svg>
          Generate Speech
        }
      </button>
    </form>
  `,
})
export class SynthesisFormComponent {
  @Input() text = '';
  @Input() error = '';
  @Input() disabled = false;
  @Input() submitting = false;
  @Input() canSubmit = false;
  @Input() sanitizeText = true;
  @Input() normalizeText = true;

  @Output() readonly textChange = new EventEmitter<string>();
  @Output() readonly sanitizeTextChange = new EventEmitter<boolean>();
  @Output() readonly normalizeTextChange = new EventEmitter<boolean>();
  @Output() readonly generate = new EventEmitter<void>();

  @ViewChild('textArea') private textArea?: ElementRef<HTMLTextAreaElement>;

  onTextInput(event: Event): void {
    this.textChange.emit((event.target as HTMLTextAreaElement).value);
  }

  onSanitizeTextChange(event: Event): void {
    this.sanitizeTextChange.emit((event.target as HTMLInputElement).checked);
  }

  onNormalizeTextChange(event: Event): void {
    this.normalizeTextChange.emit((event.target as HTMLInputElement).checked);
  }

  onSubmit(event: SubmitEvent): void {
    event.preventDefault();
    this.generate.emit();
  }

  focusText(): void {
    this.textArea?.nativeElement.focus();
  }
}
