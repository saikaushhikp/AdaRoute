/**
 * Client-side input validation for AdaRoute requests.
 * Produces user-friendly validation messages.
 */

import { RequestSettings } from './types';

export interface ValidationResult {
  isValid: boolean;
  errors: Record<string, string>;
}

export function validateSettings(settings: RequestSettings): ValidationResult {
  const errors: Record<string, string> = {};

  if (isNaN(settings.budget) || settings.budget < 0) {
    errors.budget = 'Budget must be a non-negative number ($).';
  }

  if (isNaN(settings.latencySlo) || settings.latencySlo <= 0) {
    errors.latencySlo = 'Target latency must be greater than 0 seconds.';
  }

  if (isNaN(settings.maxTokens) || settings.maxTokens < 1 || !Number.isInteger(settings.maxTokens)) {
    errors.maxTokens = 'Maximum output tokens must be a positive integer (e.g. 256).';
  }

  if (settings.gatewayUrl.trim() !== '') {
    try {
      // Allow relative paths like "" or "/api" or full URLs
      if (settings.gatewayUrl.startsWith('http://') || settings.gatewayUrl.startsWith('https://')) {
        new URL(settings.gatewayUrl);
      }
    } catch {
      errors.gatewayUrl = 'Please enter a valid URL (e.g. http://localhost:8000).';
    }
  }

  return {
    isValid: Object.keys(errors).length === 0,
    errors,
  };
}

export function validatePrompt(prompt: string): string | null {
  if (!prompt || prompt.trim().length === 0) {
    return 'Prompt cannot be empty.';
  }
  return null;
}
