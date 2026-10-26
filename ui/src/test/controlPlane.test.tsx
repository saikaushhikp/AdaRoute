import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ControlPlanePanel } from '../components/ControlPlanePanel';
import { GatewayClient } from '../lib/gatewayClient';

describe('Auditable Control Plane Component (Week-5 Milestone M12)', () => {
  it('[TC_M12_06] renders attributable savings metrics and transaction ledger', async () => {
    const mockClient = {
      getLedger: vi.fn().mockResolvedValue({
        summary: {
          total_requests: 42,
          actual_cost: 0.0052,
          baseline_cost: 0.084,
          attributable_savings: 0.0788,
          savings_percentage: 93.8,
          cache_hits: 15,
          cache_hit_rate: 35.7,
        },
        recent_transactions: [
          {
            request_id: 'req-cp-1',
            logical_model: 'general',
            selected_model: 'qwen2.5:1.5b',
            actual_cost: 0.0001,
            baseline_cost: 0.002,
            attributable_savings: 0.0019,
            cache_hit: false,
            status: 'success',
          },
        ],
      }),
      getPolicies: vi.fn().mockResolvedValue({
        active_policy: 'adaptive',
        active_version: 3,
        history: [
          { version: 1, policy_name: 'adaptive', author: 'system' },
          { version: 2, policy_name: 'cost_first', author: 'devops' },
          { version: 3, policy_name: 'adaptive', author: 'ui_operator' },
        ],
        available_policies: ['adaptive', 'cost_first', 'quality_first', 'strongest', 'cheapest'],
      }),
      rollbackPolicy: vi.fn().mockResolvedValue({
        status: 'success',
        message: "Successfully activated policy 'cost_first' at version 4",
      }),
    } as unknown as GatewayClient;

    render(
      <ControlPlanePanel
        isOpen={true}
        onClose={vi.fn()}
        gatewayClient={mockClient}
      />
    );

    // Verify title and metrics render
    expect(screen.getByText(/Auditable Control Plane & Cost Ledger/i)).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText(/\+\$0.0788/)).toBeInTheDocument();
      expect(screen.getByText(/93.8% Saved/i)).toBeInTheDocument();
      expect(screen.getByText(/\$0.0052/)).toBeInTheDocument();
    });

    // Verify policy governance section
    expect(screen.getByText(/Version 3/i)).toBeInTheDocument();

    // Verify transaction row
    expect(screen.getByText('qwen2.5:1.5b')).toBeInTheDocument();
    expect(screen.getByText(/\+\$0.00190/)).toBeInTheDocument();

    // Test rollback button click
    const select = screen.getByLabelText(/Rollback Target:/i);
    fireEvent.change(select, { target: { value: 'cost_first' } });

    const rollbackBtn = screen.getByRole('button', { name: /Rollback/i });
    expect(rollbackBtn).not.toBeDisabled();
    fireEvent.click(rollbackBtn);

    await waitFor(() => {
      expect(mockClient.rollbackPolicy).toHaveBeenCalledWith({
        policy_name: 'cost_first',
        author: 'ui_operator',
      });
    });
  });

  it('[TC_M12_07] does not render when isOpen is false', () => {
    const { container } = render(
      <ControlPlanePanel
        isOpen={false}
        onClose={vi.fn()}
      />
    );
    expect(container.firstChild).toBeNull();
  });
});
