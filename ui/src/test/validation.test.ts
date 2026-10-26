import { describe, it, expect } from 'vitest';
import { validatePrompt, validateSettings } from '../lib/validation';
import { DEFAULT_SETTINGS } from '../state/chatState';

describe('Validation Logic', () => {
  it('[TC_M1_22] validates prompts correctly', () => {
    expect(validatePrompt('')).toBe('Prompt cannot be empty.');
    expect(validatePrompt('   ')).toBe('Prompt cannot be empty.');
    expect(validatePrompt('Hello AdaRoute')).toBeNull();
  });

  it('[TC_M1_23] validates default settings as valid', () => {
    const result = validateSettings(DEFAULT_SETTINGS);
    expect(result.isValid).toBe(true);
    expect(Object.keys(result.errors).length).toBe(0);
  });

  it('[TC_M1_24] rejects negative budgets', () => {
    const result = validateSettings({
      ...DEFAULT_SETTINGS,
      budget: -0.5,
    });
    expect(result.isValid).toBe(false);
    expect(result.errors.budget).toContain('non-negative');
  });

  it('[TC_M1_25] rejects non-positive latency SLO', () => {
    const result = validateSettings({
      ...DEFAULT_SETTINGS,
      latencySlo: 0,
    });
    expect(result.isValid).toBe(false);
    expect(result.errors.latencySlo).toContain('greater than 0');
  });

  it('[TC_M1_26] rejects invalid maxTokens', () => {
    const result = validateSettings({
      ...DEFAULT_SETTINGS,
      maxTokens: 0,
    });
    expect(result.isValid).toBe(false);
    expect(result.errors.maxTokens).toContain('positive integer');
  });
});
