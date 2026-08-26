import {
  ChangeDetectionStrategy,
  Component,
  Input,
  OnChanges,
  signal,
  SimpleChanges,
} from '@angular/core';

import { SynthesisResult } from '../synthesis/synthesis.types';

@Component({
  selector: 'ovl-audio-player',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  styleUrl: './audio-player.component.css',
  template: `
    <section class="player-card" aria-labelledby="audio-player-heading">
      <div class="card-heading">
        <span>02 / OUTPUT</span>
        @if (audioUrl) {
          <span class="ready-state"><i aria-hidden="true"></i> Ready</span>
        }
      </div>
      <h3 id="audio-player-heading">Generated audio</h3>

      @if (audioUrl; as source) {
        <div class="waveform" aria-hidden="true">
          @for (bar of waveformBars; track $index) {
            <i [style.height.px]="bar"></i>
          }
        </div>
        <audio
          #audio
          controls
          preload="metadata"
          aria-label="Generated speech audio"
          [src]="source"
          (loadedmetadata)="setDuration(audio.duration)"
          (error)="playbackError.set(true)"
        >
          Your browser does not support audio playback.
        </audio>
        <div class="audio-meta">
          <span>{{ fileName }}</span>
          @if (duration() !== null) {
            <span>{{ formattedDuration }}</span>
          }
        </div>
        @if (playbackError()) {
          <p class="playback-error" role="alert">
            Audio could not be loaded. Confirm the backend is still running and generate it again.
          </p>
        }
      } @else {
        <div class="empty-output">
          <svg aria-hidden="true" viewBox="0 0 24 24" width="28" height="28">
            <path d="M4 14v-4m4 7V7m4 14V3m4 14V7m4 7v-4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" />
          </svg>
          <p>Your generated WAV will land here.</p>
          <span>Submit text to unlock playback.</span>
        </div>
      }
    </section>
  `,
})
export class AudioPlayerComponent implements OnChanges {
  @Input() result: SynthesisResult | null = null;

  readonly duration = signal<number | null>(null);
  readonly playbackError = signal(false);
  readonly waveformBars = [14, 28, 20, 40, 25, 52, 34, 44, 18, 36, 50, 24, 42, 30, 16];

  get audioUrl(): string | null {
    return this.result?.audioUrl ?? null;
  }

  get fileName(): string {
    const path = this.audioUrl ?? '';
    return decodeURIComponent(path.split('/').pop() ?? 'generated-audio.wav');
  }

  get formattedDuration(): string {
    const totalSeconds = Math.max(0, Math.round(this.duration() ?? 0));
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['result']) {
      this.duration.set(null);
      this.playbackError.set(false);
    }
  }

  setDuration(duration: number): void {
    if (Number.isFinite(duration)) {
      this.duration.set(duration);
    }
  }
}
