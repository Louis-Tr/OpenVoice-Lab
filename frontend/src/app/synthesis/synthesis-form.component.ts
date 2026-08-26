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

  @Output() readonly textChange = new EventEmitter<string>();
  @Output() readonly generate = new EventEmitter<void>();

  @ViewChild('textArea') private textArea?: ElementRef<HTMLTextAreaElement>;

  onTextInput(event: Event): void {
    this.textChange.emit((event.target as HTMLTextAreaElement).value);
  }

  onSubmit(event: SubmitEvent): void {
    event.preventDefault();
    this.generate.emit();
  }

  focusText(): void {
    this.textArea?.nativeElement.focus();
  }
}
