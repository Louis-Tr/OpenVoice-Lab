import '@angular/compiler';

import { of } from 'rxjs';
import { describe, expect, it, vi } from 'vitest';

import { ExperimentApiService } from '../api/experiment-api.service';
import { ExperimentComparisonJob, ExperimentFixture, ExperimentModelSummary } from './experiment.types';
import { LiveComparisonComponent } from './live-comparison.component';

const models: readonly ExperimentModelSummary[] = [
  {
    id: 'speecht5-pretrained',
    name: 'Pretrained',
    role: 'pretrained',
    variant: 'pretrained',
    runtime: 'PyTorch CPU',
    hosting: 'self-hosted',
    revision: 'revision',
    modelSha256: 'a'.repeat(64),
    available: true,
    unavailableReason: null,
  },
  {
    id: 'speecht5-v3-replay',
    name: 'V3 Replay',
    role: 'adapted',
    variant: 'v3-replay',
    runtime: 'PyTorch CPU',
    hosting: 'self-hosted',
    revision: 'revision',
    modelSha256: 'b'.repeat(64),
    available: true,
    unavailableReason: null,
  },
];

const fixture: ExperimentFixture = {
  id: 'fixture-1',
  text: 'The patient has arm pain.',
  targetTerms: [{ text: 'arm', canonical: 'arm', category: 'anatomy' }],
};

const queued: ExperimentComparisonJob = {
  id: 'comparison-1',
  mode: 'fixture',
  stage: 'queued',
  progressPercent: 0,
  originalText: fixture.text,
  normalizedText: null,
  targetTerms: ['arm'],
  sanitizeText: true,
  normalizeText: true,
  modelIds: ['speecht5-pretrained', 'speecht5-v3-replay'],
  results: [],
  createdAt: '2026-08-27T00:00:00Z',
  updatedAt: '2026-08-27T00:00:00Z',
  completedAt: null,
  error: null,
};

function api(): ExperimentApiService {
  return {
    startComparison: vi.fn(() => of(queued)),
    getComparison: vi.fn(() => of({ ...queued, stage: 'completed', progressPercent: 100 })),
    cancelComparison: vi.fn(() => of({ ...queued, stage: 'cancelled' })),
    comparisonEventsUrl: vi.fn(() => '/events'),
  } as unknown as ExperimentApiService;
}

describe('LiveComparisonComponent', () => {
  it('sends independent cleanup options with a locked fixture comparison', () => {
    const service = api();
    const component = new LiveComparisonComponent(service);
    component.models = models;
    component.fixtures = [fixture];
    component.sanitizeText.set(false);
    component.normalizeText.set(true);

    component.run(new Event('submit'));

    expect(service.startComparison).toHaveBeenCalledWith({
      mode: 'fixture',
      fixtureId: fixture.id,
      modelIds: ['speecht5-pretrained', 'speecht5-v3-replay'],
      sanitizeText: false,
      normalizeText: true,
    });
    component.ngOnDestroy();
  });

  it('rejects a custom target term that is absent from the text', () => {
    const service = api();
    const component = new LiveComparisonComponent(service);
    component.models = models;
    component.fixtures = [fixture];
    component.mode.set('custom');
    component.customText.set('The patient is stable.');
    component.customTerms.set('amlodipine');

    component.run(new Event('submit'));

    expect(component.validationError()).toContain('must appear');
    expect(service.startComparison).not.toHaveBeenCalled();
  });

  it('keeps the pretrained model identifiable as the live control', () => {
    const component = new LiveComparisonComponent(api());
    component.models = models;

    expect(component.modelFor('speecht5-pretrained')?.role).toBe('pretrained');
    expect(component.selectedModels()).toContain('speecht5-pretrained');
    expect(component.hasScoredResults(queued)).toBe(false);
  });
});
