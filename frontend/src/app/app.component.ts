import { ChangeDetectionStrategy, Component } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';

@Component({
  selector: 'ovl-root',
  standalone: true,
  imports: [RouterLink, RouterLinkActive, RouterOutlet],
  changeDetection: ChangeDetectionStrategy.OnPush,
  styleUrl: './app.component.css',
  template: `
    <a class="skip-link" href="#main-content">Skip to synthesis</a>
    <header class="app-header">
      <a class="brand" routerLink="/" aria-label="OpenVoice Lab home">
        <span class="brand-mark" aria-hidden="true">OV</span>
        <h1>OpenVoice Lab</h1>
      </a>
      <nav aria-label="Primary navigation">
        <a routerLink="/" routerLinkActive="active" [routerLinkActiveOptions]="{ exact: true }">
          Synthesis
        </a>
        <a routerLink="/benchmarks" routerLinkActive="active">Benchmarks</a>
      </nav>
    </header>
    <main id="main-content">
      <router-outlet />
    </main>
  `,
})
export class AppComponent {}
