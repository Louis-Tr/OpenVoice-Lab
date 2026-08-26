import { ChangeDetectionStrategy, Component } from '@angular/core';

@Component({
  selector: 'ovl-audio-player',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <section aria-labelledby="audio-player-heading">
      <h3 id="audio-player-heading">Audio</h3>
      <p>Playback and duration will appear after a synthesis result is available.</p>
    </section>
  `,
})
export class AudioPlayerComponent {}

