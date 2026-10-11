import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { GatewayStatus } from '../components/GatewayStatus';
import { MessageBubble } from '../components/MessageBubble';
import { Composer } from '../components/Composer';
import { SettingsPanel } from '../components/SettingsPanel';
import { DEFAULT_SETTINGS } from '../state/chatState';

describe('UI Presentation Components', () => {
  it('renders GatewayStatus as active when healthy', () => {
    const onRefresh = vi.fn();
    render(
      <GatewayStatus
        health={{ status: 'healthy', replicas: ['local-r1'] }}
        onRefresh={onRefresh}
      />
    );

    expect(screen.getByText('Gateway Active')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /refresh gateway health/i }));
    expect(onRefresh).toHaveBeenCalled();
  });

  it('renders GatewayStatus as offline when unavailable', () => {
    render(
      <GatewayStatus
        health={{ status: 'unavailable', error: 'Connection refused' }}
        onRefresh={vi.fn()}
      />
    );

    expect(screen.getByText('Gateway Offline')).toBeInTheDocument();
  });

  it('renders user and assistant MessageBubble', () => {
    const userMsg = {
      id: '1',
      role: 'user' as const,
      content: 'Can you help me route?',
      timestamp: Date.now(),
    };

    const { rerender } = render(<MessageBubble message={userMsg} />);
    expect(screen.getByText('Can you help me route?')).toBeInTheDocument();
    expect(screen.getByText('You')).toBeInTheDocument();

    const assistantMsg = {
      id: '2',
      role: 'assistant' as const,
      content: 'Routing decision computed.',
      timestamp: Date.now(),
      diagnostics: {
        request_id: 'req-1',
        logical_model: 'general',
        selected_model: 'qwen2.5:1.5b',
        selected_replica: 'local-r1',
        provider: 'local',
        latency_ms: 50.0,
        attempts: 1,
        cache_hit: false,
        estimated_cost: 0,
        actual_cost: 0,
        slo_attained: true,
      },
    };

    rerender(<MessageBubble message={assistantMsg} onInspectDiagnostics={vi.fn()} />);
    expect(screen.getByText('Routing decision computed.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /diagnostics/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /copy/i })).toBeInTheDocument();
  });

  it('renders Composer and triggers onSend on button click and keyboard Enter', () => {
    const onSend = vi.fn();
    const onChange = vi.fn();

    const { rerender } = render(
      <Composer
        value=""
        onChange={onChange}
        onSend={onSend}
        isLoading={false}
      />
    );

    const sendBtn = screen.getByRole('button', { name: /send message/i });
    expect(sendBtn).toBeDisabled();

    rerender(
      <Composer
        value="Hello AdaRoute"
        onChange={onChange}
        onSend={onSend}
        isLoading={false}
      />
    );

    expect(sendBtn).not.toBeDisabled();
    fireEvent.click(sendBtn);
    expect(onSend).toHaveBeenCalledTimes(1);

    const textarea = screen.getByPlaceholderText(/ask a question/i);
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false });
    expect(onSend).toHaveBeenCalledTimes(2);

    // Shift+Enter should not trigger send
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: true });
    expect(onSend).toHaveBeenCalledTimes(2);
  });

  it('allows changing capability and privacy in SettingsPanel', () => {
    const onChange = vi.fn();
    render(
      <SettingsPanel
        settings={DEFAULT_SETTINGS}
        onChange={onChange}
        errors={{}}
        onReset={vi.fn()}
      />
    );

    // Select Coding capability
    const codingBtn = screen.getByRole('radio', { name: /coding/i });
    fireEvent.click(codingBtn);
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({ capability: 'coding' })
    );

    // Change privacy level
    const select = screen.getByLabelText(/privacy level/i);
    fireEvent.change(select, { target: { value: 'internal_only' } });
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({ privacyLevel: 'internal_only' })
    );
  });
});
