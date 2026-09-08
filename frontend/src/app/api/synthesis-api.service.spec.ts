import '@angular/compiler';

import { HttpClient } from '@angular/common/http';
import { of } from 'rxjs';
import { expect, it, vi } from 'vitest';

import { SynthesisApiService } from './synthesis-api.service';

it('sends idempotency keys on job creation and polls the encoded job URL', () => {
  const http = { post: vi.fn(() => of({})), get: vi.fn(() => of({})) };
  const api = new SynthesisApiService(http as unknown as HttpClient, '/api');
  const request = {
    text: 'Hello', modelId: 'kokoro-fp32', voiceId: 'af_heart',
    sanitizeText: true, normalizeText: false,
  };
  api.startJob(request, 'stable-key').subscribe();
  expect(http.post).toHaveBeenCalledExactlyOnceWith('/api/synthesis/jobs', request, {
    headers: { 'Idempotency-Key': 'stable-key' },
  });
  api.getJob('job/a').subscribe();
  expect(http.get).toHaveBeenCalledExactlyOnceWith('/api/synthesis/jobs/job%2Fa');
});
