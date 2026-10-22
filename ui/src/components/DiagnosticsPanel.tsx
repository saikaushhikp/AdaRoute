import React from 'react';
import { RoutingMetadata } from '../lib/types';
import { X, Cpu, Server, Zap, Clock, ShieldCheck, DollarSign } from 'lucide-react';

interface DiagnosticsPanelProps {
  isOpen: boolean;
  onClose: () => void;
  metadata: RoutingMetadata | null;
}

export const DiagnosticsPanel: React.FC<DiagnosticsPanelProps> = ({
  isOpen,
  onClose,
  metadata,
}) => {
  return (
    <>
      <div
        className={`drawer-backdrop ${isOpen ? 'open' : ''}`}
        onClick={onClose}
        aria-hidden="true"
      />
      <aside
        className={`diagnostics-drawer ${isOpen ? 'open' : ''}`}
        aria-label="Routing Diagnostics"
        role="dialog"
        aria-modal="true"
      >
        <div className="drawer-header">
          <h3>
            <Zap size={18} color="var(--accent-secondary)" />
            Routing Diagnostics
          </h3>
          <button
            className="drawer-close-btn"
            onClick={onClose}
            aria-label="Close diagnostics panel"
            type="button"
          >
            <X size={18} />
          </button>
        </div>

        <div className="drawer-content">
          {!metadata ? (
            <div style={{ textAlign: 'center', color: 'var(--text-dim)', padding: '2rem 1rem' }}>
              <p>No routing telemetry available yet.</p>
              <p style={{ fontSize: '0.8rem', marginTop: '0.5rem' }}>
                Send a message to inspect live gateway dispatch metadata.
              </p>
            </div>
          ) : (
            <>
              {/* Summary Status */}
              <div className="diag-card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                  <span className="diag-label">Execution Status</span>
                  <span className={`diag-badge ${metadata.cache_hit ? 'hit' : 'miss'}`}>
                    {metadata.cache_hit ? 'CACHE HIT' : 'CACHE MISS'}
                  </span>
                </div>
                <div className="diag-item">
                  <span className="diag-label">Request Correlation ID</span>
                  <span className="diag-value" style={{ fontSize: '0.78rem' }}>
                    {metadata.request_id}
                  </span>
                </div>
              </div>

              {/* Model & Routing Decision */}
              <div className="diag-card">
                <h4 style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <Cpu size={14} /> Stage 1: Model Selection
                </h4>
                <div className="diag-grid">
                  <div className="diag-item">
                    <span className="diag-label">Requested Logical</span>
                    <span className="diag-value highlight">{metadata.logical_model}</span>
                  </div>
                  <div className="diag-item">
                    <span className="diag-label">Selected Model</span>
                    <span className="diag-value">{metadata.selected_model}</span>
                  </div>
                  <div className="diag-item">
                    <span className="diag-label">Routing Policy</span>
                    <span className="diag-value">{metadata.policy_name || 'adaptive'}</span>
                  </div>
                  <div className="diag-item">
                    <span className="diag-label">Provider</span>
                    <span className="diag-value">{metadata.provider}</span>
                  </div>
                </div>
              </div>

              {/* Replica & Scheduling */}
              <div className="diag-card">
                <h4 style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <Server size={14} /> Stage 2: Replica & Scheduling
                </h4>
                <div className="diag-grid">
                  <div className="diag-item">
                    <span className="diag-label">Replica ID</span>
                    <span className="diag-value highlight">{metadata.selected_replica}</span>
                  </div>
                  <div className="diag-item">
                    <span className="diag-label">Attempt Count</span>
                    <span className="diag-value">{metadata.attempts}</span>
                  </div>
                </div>
              </div>

              {/* Telemetry & SLA */}
              <div className="diag-card">
                <h4 style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <Clock size={14} /> Performance & Telemetry
                </h4>
                <div className="diag-grid">
                  <div className="diag-item">
                    <span className="diag-label">Latency</span>
                    <span className="diag-value">{metadata.latency_ms} ms</span>
                  </div>
                  <div className="diag-item">
                    <span className="diag-label">SLO Attainment</span>
                    <span
                      className="diag-value"
                      style={{ color: metadata.slo_attained ? 'var(--success)' : 'var(--danger)' }}
                    >
                      {metadata.slo_attained ? 'Attained' : 'Breached'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Economic Accounting */}
              <div className="diag-card">
                <h4 style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <DollarSign size={14} /> Cost Accounting
                </h4>
                <div className="diag-grid">
                  <div className="diag-item">
                    <span className="diag-label">Estimated Cost</span>
                    <span className="diag-value">${metadata.estimated_cost.toFixed(6)}</span>
                  </div>
                  <div className="diag-item">
                    <span className="diag-label">Actual Cost</span>
                    <span className="diag-value highlight">${metadata.actual_cost.toFixed(6)}</span>
                  </div>
                </div>
              </div>

              {/* Attempt History & Failovers */}
              {metadata.attempt_history && metadata.attempt_history.length > 0 && (
                <div className="diag-card">
                  <h4 style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <Server size={14} /> Attempt History & Failovers ({metadata.attempt_history.length})
                  </h4>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    {metadata.attempt_history.map((att) => (
                      <div
                        key={att.attempt_number}
                        style={{
                          fontSize: '0.75rem',
                          padding: '0.4rem 0.6rem',
                          borderRadius: '6px',
                          backgroundColor: 'rgba(255, 255, 255, 0.04)',
                          borderLeft: `3px solid ${att.status === 'success' ? 'var(--success)' : 'var(--danger)'}`,
                        }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontWeight: 600 }}>
                          <span>Attempt #{att.attempt_number}: {att.model_id}</span>
                          <span style={{ color: att.status === 'success' ? 'var(--success)' : 'var(--danger)' }}>
                            {att.status.toUpperCase()}
                          </span>
                        </div>
                        <div style={{ display: 'flex', gap: '0.8rem', color: 'var(--text-dim)', marginTop: '0.2rem' }}>
                          <span>Replica: {att.replica_id}</span>
                          <span>Latency: {att.latency_ms.toFixed(1)}ms</span>
                          <span>Cost: ${att.cost.toFixed(6)} ({att.cost_state || 'unknown'})</span>
                        </div>
                        {att.error_message && (
                          <div style={{ color: 'var(--danger)', marginTop: '0.2rem', wordBreak: 'break-word', fontSize: '0.7rem' }}>
                            {att.error_message}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-dim)', fontSize: '0.72rem', marginTop: 'auto', padding: '0.5rem 0' }}>
                <ShieldCheck size={14} />
                <span>Zero-leakage telemetry: Raw prompts and keys are scrubbed.</span>
              </div>
            </>
          )}
        </div>
      </aside>
    </>
  );
};
