import React, { useState, useEffect, useCallback } from 'react';
import { X, RefreshCw, ShieldCheck, DollarSign, ArrowDownRight, History, Layers } from 'lucide-react';
import { defaultGatewayClient, GatewayClient } from '../lib/gatewayClient';

interface ControlPlanePanelProps {
  isOpen: boolean;
  onClose: () => void;
  gatewayClient?: GatewayClient;
}

export const ControlPlanePanel: React.FC<ControlPlanePanelProps> = ({
  isOpen,
  onClose,
  gatewayClient = defaultGatewayClient,
}) => {
  const [ledger, setLedger] = useState<any>(null);
  const [policies, setPolicies] = useState<any>(null);
  const [selectedPolicy, setSelectedPolicy] = useState<string>('adaptive');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isRollingBack, setIsRollingBack] = useState<boolean>(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setStatusMessage(null);
    try {
      const [ledgerRes, policyRes] = await Promise.all([
        gatewayClient.getLedger(30),
        gatewayClient.getPolicies(),
      ]);
      setLedger(ledgerRes);
      setPolicies(policyRes);
      if (policyRes?.active_policy) {
        setSelectedPolicy(policyRes.active_policy);
      }
    } catch (err: unknown) {
      setStatusMessage(`Error loading control plane data: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setIsLoading(false);
    }
  }, [gatewayClient]);

  useEffect(() => {
    if (isOpen) {
      fetchData();
    }
  }, [isOpen, fetchData]);

  const handleRollback = async () => {
    if (!selectedPolicy) return;
    setIsRollingBack(true);
    setStatusMessage(null);
    try {
      const res = await gatewayClient.rollbackPolicy({
        policy_name: selectedPolicy,
        author: 'ui_operator',
      });
      setStatusMessage(res.message || `Switched active policy to ${selectedPolicy}`);
      await fetchData();
    } catch (err: unknown) {
      setStatusMessage(`Rollback failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setIsRollingBack(false);
    }
  };

  if (!isOpen) return null;

  const summary = ledger?.summary || {
    total_requests: 0,
    actual_cost: 0,
    baseline_cost: 0,
    attributable_savings: 0,
    savings_percentage: 0,
    cache_hits: 0,
    cache_hit_rate: 0,
  };

  return (
    <>
      <div className="drawer-backdrop open" onClick={onClose} aria-hidden="true" />
      <aside
        className="diagnostics-drawer open"
        aria-label="Auditable Control Plane"
        role="dialog"
        aria-modal="true"
        style={{ width: 'min(90vw, 680px)' }}
      >
        <div className="drawer-header">
          <h3>
            <ShieldCheck size={18} color="var(--accent-primary)" />
            Auditable Control Plane & Cost Ledger
          </h3>
          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
            <button
              className="action-btn"
              onClick={fetchData}
              disabled={isLoading}
              title="Refresh ledger telemetry"
              type="button"
            >
              <RefreshCw size={14} className={isLoading ? 'spin' : ''} />
              <span>Refresh</span>
            </button>
            <button
              className="drawer-close-btn"
              onClick={onClose}
              aria-label="Close control plane"
              type="button"
            >
              <X size={18} />
            </button>
          </div>
        </div>

        <div className="drawer-content" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {statusMessage && (
            <div
              style={{
                padding: '0.6rem 0.8rem',
                borderRadius: '6px',
                background: statusMessage.includes('failed') || statusMessage.includes('Error')
                  ? 'rgba(239, 68, 68, 0.15)'
                  : 'rgba(16, 185, 129, 0.15)',
                color: statusMessage.includes('failed') || statusMessage.includes('Error')
                  ? '#ef4444'
                  : '#10b981',
                fontSize: '0.82rem',
                border: '1px solid currentColor',
              }}
            >
              {statusMessage}
            </div>
          )}

          {/* Attributable Cost Metric Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '0.75rem' }}>
            <div className="diag-card" style={{ borderLeft: '3px solid var(--accent-primary)' }}>
              <div className="diag-label" style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                <DollarSign size={13} /> Attributable Savings (S_p)
              </div>
              <div style={{ fontSize: '1.15rem', fontWeight: 700, color: '#10b981', marginTop: '0.3rem' }}>
                +${summary.attributable_savings.toFixed(4)}
              </div>
              <span className="diag-badge hit" style={{ marginTop: '0.3rem', display: 'inline-block' }}>
                {summary.savings_percentage}% Saved
              </span>
            </div>

            <div className="diag-card">
              <div className="diag-label">Actual Cost (A_p)</div>
              <div style={{ fontSize: '1.15rem', fontWeight: 700, marginTop: '0.3rem' }}>
                ${summary.actual_cost.toFixed(4)}
              </div>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)', marginTop: '0.3rem', display: 'block' }}>
                {summary.total_requests} Requests
              </span>
            </div>

            <div className="diag-card">
              <div className="diag-label">Baseline Cost (B_p)</div>
              <div style={{ fontSize: '1.15rem', fontWeight: 700, color: 'var(--text-muted)', marginTop: '0.3rem' }}>
                ${summary.baseline_cost.toFixed(4)}
              </div>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)', marginTop: '0.3rem', display: 'block' }}>
                Frontier Pricing
              </span>
            </div>

            <div className="diag-card">
              <div className="diag-label">Cache Efficiency</div>
              <div style={{ fontSize: '1.15rem', fontWeight: 700, color: 'var(--accent-secondary)', marginTop: '0.3rem' }}>
                {summary.cache_hit_rate}%
              </div>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)', marginTop: '0.3rem', display: 'block' }}>
                {summary.cache_hits} Cache Hits
              </span>
            </div>
          </div>

          {/* Versioned Policy Governance & Rollback */}
          <div className="diag-card">
            <h4 style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Layers size={15} /> Versioned Policy Governance & Rollback
            </h4>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem', alignItems: 'center' }}>
              <div style={{ flex: '1 1 200px' }}>
                <span className="diag-label">Active Gateway Policy:</span>
                <div style={{ marginTop: '0.3rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span className="diag-badge hit" style={{ textTransform: 'uppercase' }}>
                    {policies?.active_policy || 'adaptive'}
                  </span>
                  <span style={{ fontSize: '0.78rem', color: 'var(--text-dim)' }}>
                    Version {policies?.active_version || 1}
                  </span>
                </div>
              </div>

              <div style={{ flex: '1 1 200px', display: 'flex', gap: '0.5rem', alignItems: 'flex-end' }}>
                <div style={{ flex: 1 }}>
                  <label htmlFor="policy-select" className="diag-label" style={{ display: 'block', marginBottom: '0.25rem' }}>
                    Rollback Target:
                  </label>
                  <select
                    id="policy-select"
                    value={selectedPolicy}
                    onChange={(e) => setSelectedPolicy(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '0.4rem 0.5rem',
                      background: 'var(--bg-input)',
                      border: '1px solid var(--border-color)',
                      color: 'var(--text-primary)',
                      borderRadius: '4px',
                      fontSize: '0.82rem',
                    }}
                  >
                    {(policies?.available_policies || ['adaptive', 'cost_first', 'quality_first', 'strongest', 'cheapest']).map(
                      (p: string) => (
                        <option key={p} value={p}>
                          {p}
                        </option>
                      )
                    )}
                  </select>
                </div>
                <button
                  type="button"
                  className="action-btn"
                  onClick={handleRollback}
                  disabled={isRollingBack || selectedPolicy === policies?.active_policy}
                  style={{
                    background: 'var(--accent-primary)',
                    color: '#fff',
                    padding: '0.42rem 0.75rem',
                    borderRadius: '4px',
                    fontWeight: 600,
                  }}
                >
                  <ArrowDownRight size={14} />
                  <span>{isRollingBack ? 'Applying...' : 'Rollback'}</span>
                </button>
              </div>
            </div>

            {/* Version History Audit Log */}
            {policies?.history && policies.history.length > 0 && (
              <div style={{ marginTop: '0.85rem', borderTop: '1px solid var(--border-color)', paddingTop: '0.6rem' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginBottom: '0.4rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                  <History size={12} /> Audit History (Latest {policies.history.length} versions)
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem', maxHeight: '110px', overflowY: 'auto' }}>
                  {policies.history.slice().reverse().map((h: any) => (
                    <div
                      key={h.version}
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        fontSize: '0.74rem',
                        padding: '0.2rem 0.4rem',
                        background: h.version === policies.active_version ? 'rgba(59, 130, 246, 0.1)' : 'transparent',
                        borderRadius: '3px',
                      }}
                    >
                      <span>
                        <strong>v{h.version}</strong>: {h.policy_name} ({h.description || 'updated'})
                      </span>
                      <span style={{ color: 'var(--text-dim)' }}>by {h.author || 'system'}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Attributable Transactions Table */}
          <div className="diag-card">
            <h4 style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.6rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <DollarSign size={14} /> Attributable Transaction Ledger
            </h4>
            {(!ledger?.recent_transactions || ledger.recent_transactions.length === 0) ? (
              <div style={{ textAlign: 'center', color: 'var(--text-dim)', padding: '1rem', fontSize: '0.8rem' }}>
                No ledger transactions recorded yet.
              </div>
            ) : (
              <div style={{ overflowX: 'auto', maxHeight: '220px', overflowY: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.75rem', textAlign: 'left' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--border-color)', color: 'var(--text-dim)' }}>
                      <th style={{ padding: '0.35rem' }}>Capability</th>
                      <th style={{ padding: '0.35rem' }}>Model</th>
                      <th style={{ padding: '0.35rem' }}>Actual (A_p)</th>
                      <th style={{ padding: '0.35rem' }}>Baseline (B_p)</th>
                      <th style={{ padding: '0.35rem' }}>Savings (S_p)</th>
                      <th style={{ padding: '0.35rem' }}>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {ledger.recent_transactions.map((tx: any, idx: number) => (
                      <tr key={tx.request_id || idx} style={{ borderBottom: '1px solid var(--border-color)' }}>
                        <td style={{ padding: '0.35rem', fontWeight: 600 }}>{tx.logical_model}</td>
                        <td style={{ padding: '0.35rem', color: 'var(--text-muted)' }}>{tx.selected_model || 'none'}</td>
                        <td style={{ padding: '0.35rem' }}>${Number(tx.actual_cost).toFixed(5)}</td>
                        <td style={{ padding: '0.35rem', color: 'var(--text-dim)' }}>${Number(tx.baseline_cost).toFixed(5)}</td>
                        <td style={{ padding: '0.35rem', color: '#10b981', fontWeight: 600 }}>
                          +${Number(tx.attributable_savings).toFixed(5)}
                        </td>
                        <td style={{ padding: '0.35rem' }}>
                          <span className={`diag-badge ${tx.cache_hit ? 'hit' : 'miss'}`} style={{ fontSize: '0.65rem' }}>
                            {tx.cache_hit ? 'CACHE' : tx.status?.toUpperCase() || 'OK'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      </aside>
    </>
  );
};
