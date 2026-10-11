/**
 * HTTP Client for AdaRoute Unified AI Gateway.
 * Manages request formation, defensive parsing, error mapping, and health checking.
 */

import {
  ChatCompletionResponse,
  ChatMessage,
  ClientError,
  GatewayHealth,
  RequestSettings,
} from './types';

export class GatewayClient {
  private baseUrl: string;

  constructor(baseUrl: string = '') {
    // Strip trailing slash if present
    this.baseUrl = baseUrl.replace(/\/+$/, '');
  }

  public setBaseUrl(url: string) {
    this.baseUrl = url.replace(/\/+$/, '');
  }

  public getBaseUrl(): string {
    return this.baseUrl;
  }

  private resolveUrl(path: string): string {
    const cleanPath = path.startsWith('/') ? path : `/${path}`;
    if (!this.baseUrl) {
      return cleanPath;
    }
    return `${this.baseUrl}${cleanPath}`;
  }

  /**
   * Health check against GET /health
   */
  async checkHealth(signal?: AbortSignal): Promise<GatewayHealth> {
    const url = this.resolveUrl('/health');
    try {
      const response = await fetch(url, {
        method: 'GET',
        headers: { Accept: 'application/json' },
        signal: signal || AbortSignal.timeout(5000),
      });

      if (!response.ok) {
        return {
          status: 'unavailable',
          error: `Gateway returned status ${response.status}: ${response.statusText}`,
          checkedAt: Date.now(),
        };
      }

      const data = await response.json();
      return {
        status: data.status === 'healthy' ? 'healthy' : 'unavailable',
        models: Array.isArray(data.models) ? data.models : [],
        replicas: Array.isArray(data.replicas) ? data.replicas : [],
        checkedAt: Date.now(),
      };
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Connection failed';
      return {
        status: 'unavailable',
        error: message,
        checkedAt: Date.now(),
      };
    }
  }

  /**
   * Dispatches chat completion to POST /v1/chat/completions
   */
  async sendChatCompletion(
    messages: ChatMessage[],
    settings: RequestSettings,
    signal?: AbortSignal
  ): Promise<ChatCompletionResponse> {
    const url = this.resolveUrl('/v1/chat/completions');

    // Build payload preserving logical capability and constraints
    const payload: Record<string, unknown> = {
      model: settings.capability,
      messages: messages.map((m) => ({ role: m.role, content: m.content })),
      privacy_level: settings.privacyLevel,
      budget: settings.budget,
      quality_requirement: settings.qualityRequirement,
      latency_slo: settings.latencySlo,
      max_tokens: settings.maxTokens,
      tenant_id: settings.tenantId || 'default',
      user_id: settings.userId || 'anonymous',
      policy: settings.policy,
      scheduler: settings.scheduler,
      no_cache: settings.noCache,
    };

    let response: Response;
    try {
      response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
        body: JSON.stringify(payload),
        signal,
      });
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        const error: ClientError = {
          code: 'TIMEOUT',
          message: 'Request was cancelled or timed out.',
        };
        throw error;
      }
      const error: ClientError = {
        code: 'NETWORK_ERROR',
        message: 'Could not connect to AdaRoute Gateway. Please check your network and gateway status.',
        technicalDetail: err instanceof Error ? err.message : String(err),
      };
      throw error;
    }

    if (!response.ok) {
      let errorDetail = response.statusText;
      try {
        const errJson = await response.json();
        if (errJson && typeof errJson === 'object' && errJson.detail) {
          errorDetail = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
        }
      } catch {
        // Response body not JSON
      }

      if (response.status === 400) {
        const error: ClientError = {
          status: 400,
          code: 'VALIDATION_ERROR',
          message: `Request validation failed: ${errorDetail}`,
          technicalDetail: errorDetail,
        };
        throw error;
      }

      if (response.status === 422) {
        const error: ClientError = {
          status: 422,
          code: 'VALIDATION_ERROR',
          message: `Malformed request parameters: ${errorDetail}`,
          technicalDetail: errorDetail,
        };
        throw error;
      }

      if (response.status === 503) {
        const error: ClientError = {
          status: 503,
          code: 'ROUTING_REJECTED',
          message: `Routing rejected: ${errorDetail}`,
          technicalDetail: errorDetail,
        };
        throw error;
      }

      const error: ClientError = {
        status: response.status,
        code: 'SERVER_ERROR',
        message: `Gateway returned unexpected error (${response.status}): ${errorDetail}`,
        technicalDetail: errorDetail,
      };
      throw error;
    }

    try {
      const data = (await response.json()) as ChatCompletionResponse;
      // Ensure content field is normalized
      if (!data.content && data.choices && data.choices.length > 0 && data.choices[0].message) {
        data.content = data.choices[0].message.content;
      }
      return data;
    } catch (parseErr: unknown) {
      const error: ClientError = {
        code: 'UNKNOWN',
        message: 'Received invalid or unparseable response from AdaRoute.',
        technicalDetail: parseErr instanceof Error ? parseErr.message : String(parseErr),
      };
      throw error;
    }
  }
}

export const defaultGatewayClient = new GatewayClient();
