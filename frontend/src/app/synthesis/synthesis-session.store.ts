import { Inject, Injectable, InjectionToken } from '@angular/core';

import { API_BASE_URL } from '../core/api-base-url.token';
import { SynthesisRequest } from './synthesis.types';

export interface SynthesisSession {
  readonly version: 1;
  readonly key: string;
  readonly request: SynthesisRequest;
  readonly jobId: string | null;
  readonly createdAt: number;
  readonly stoppedReason: string | null;
}

export const SYNTHESIS_STORAGE = new InjectionToken<Storage | null>('SYNTHESIS_STORAGE', {
  providedIn: 'root',
  factory: () => {
    try {
      return globalThis.localStorage ?? null;
    } catch {
      return null;
    }
  },
});

function isRequest(value: unknown): value is SynthesisRequest {
  if (!value || typeof value !== 'object') return false;
  const request = value as Partial<SynthesisRequest>;
  return typeof request.text === 'string' && request.text.length > 0 && request.text.length <= 5000
    && typeof request.modelId === 'string' && request.modelId.length > 0
    && typeof request.voiceId === 'string' && request.voiceId.length > 0
    && typeof request.sanitizeText === 'boolean' && typeof request.normalizeText === 'boolean';
}

@Injectable({ providedIn: 'root' })
export class SynthesisSessionStore {
  private readonly storageKey: string;

  constructor(
    @Inject(SYNTHESIS_STORAGE) private readonly storage: Storage | null,
    @Inject(API_BASE_URL) apiBaseUrl: string,
  ) {
    this.storageKey = `openvoice.synthesis.v1:${apiBaseUrl}`;
  }

  load(): SynthesisSession | null {
    try {
      const serialized = this.storage?.getItem(this.storageKey);
      if (!serialized) return null;
      const value: unknown = JSON.parse(serialized);
      if (!value || typeof value !== 'object') return null;
      const session = value as Partial<SynthesisSession>;
      if (session.version !== 1 || typeof session.key !== 'string'
        || !/^[A-Za-z0-9._:-]{1,128}$/.test(session.key) || !isRequest(session.request)
        || !(session.jobId === null || (typeof session.jobId === 'string' && session.jobId.length > 0))
        || typeof session.createdAt !== 'number' || !Number.isFinite(session.createdAt)
        || !(session.stoppedReason === null || typeof session.stoppedReason === 'string')) return null;
      return session as SynthesisSession;
    } catch {
      return null;
    }
  }

  save(session: SynthesisSession): boolean {
    try {
      if (!this.storage) return false;
      this.storage.setItem(this.storageKey, JSON.stringify(session));
      return true;
    } catch {
      return false;
    }
  }
}
