import { describe, it, expect, vi, beforeEach } from 'vitest';
import { GatewayClient } from '../lib/gatewayClient';
import { DEFAULT_SETTINGS } from '../state/chatState';

describe('GatewayClient', () => {
  let client: GatewayClient;

  beforeEach(() => {
    client = new GatewayClient('http://localhost:8000');
    vi.restoreAllMocks();
  });

  it('[TC_M1_11] checks health successfully', async () => {
    const mockHealth = {
      status: 'healthy',
      models: ['qwen2.5:1.5b', 'gpt-4o-external'],
      replicas: ['local-r1', 'external-r1'],
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockHealth,
    } as Response);

    const res = await client.checkHealth();
    expect(res.status).toBe('healthy');
    expect(res.models).toContain('qwen2.5:1.5b');
    expect(res.replicas).toContain('local-r1');
  });

  it('[TC_M1_12] handles offline health status', async () => {
    global.fetch = vi.fn().mockRejectedValue(new Error('Failed to fetch'));

    const res = await client.checkHealth();
    expect(res.status).toBe('unavailable');
    expect(res.error).toContain('Failed to fetch');
  });

  it('[TC_M1_13] sends chat completion request and parses response', async () => {
    const mockResponse = {
      id: 'req-12345',
      model: 'qwen2.5:1.5b',
      choices: [
        {
          message: {
            role: 'assistant',
            content: 'Hello from AdaRoute local model!',
          },
        },
      ],
      routing_metadata: {
        request_id: 'req-12345',
        logical_model: 'general',
        selected_model: 'qwen2.5:1.5b',
        selected_replica: 'local-r1',
        provider: 'local',
        latency_ms: 120.5,
        attempts: 1,
        cache_hit: false,
        estimated_cost: 0.0,
        actual_cost: 0.0,
        slo_attained: true,
        policy_name: 'adaptive',
      },
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    } as Response);

    const res = await client.sendChatCompletion(
      [{ role: 'user', content: 'Hi' }],
      DEFAULT_SETTINGS
    );

    expect(res.id).toBe('req-12345');
    expect(res.content).toBe('Hello from AdaRoute local model!');
    expect(res.routing_metadata?.selected_replica).toBe('local-r1');
    expect(res.routing_metadata?.cache_hit).toBe(false);
  });

  it('[TC_M1_14] maps HTTP 400 to VALIDATION_ERROR', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 400,
      statusText: 'Bad Request',
      json: async () => ({ detail: 'Budget must be non-negative' }),
    } as Response);

    await expect(
      client.sendChatCompletion([{ role: 'user', content: 'Hi' }], DEFAULT_SETTINGS)
    ).rejects.toMatchObject({
      status: 400,
      code: 'VALIDATION_ERROR',
      message: expect.stringContaining('Budget must be non-negative'),
    });
  });

  it('[TC_M1_15] maps HTTP 503 to ROUTING_REJECTED', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      statusText: 'Service Unavailable',
      json: async () => ({ detail: 'No eligible models found under budget constraint' }),
    } as Response);

    await expect(
      client.sendChatCompletion([{ role: 'user', content: 'Hi' }], DEFAULT_SETTINGS)
    ).rejects.toMatchObject({
      status: 503,
      code: 'ROUTING_REJECTED',
      message: expect.stringContaining('No eligible models found'),
    });
  });

  it('[TC_M1_16] maps network errors to NETWORK_ERROR', async () => {
    global.fetch = vi.fn().mockRejectedValue(new Error('Network connection refused'));

    await expect(
      client.sendChatCompletion([{ role: 'user', content: 'Hi' }], DEFAULT_SETTINGS)
    ).rejects.toMatchObject({
      code: 'NETWORK_ERROR',
      message: expect.stringContaining('Could not connect to AdaRoute Gateway'),
    });
  });
});
