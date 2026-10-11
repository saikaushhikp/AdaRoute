import React, { useState } from 'react';
import {
  LogicalCapability,
  PrivacyLevel,
  QualityRequirement,
  ReplicaScheduler,
  RequestSettings,
  RoutingPolicy,
} from '../lib/types';
import {
  Sliders,
  ChevronDown,
  ChevronRight,
  Shield,
  RotateCcw,
  Zap,
} from 'lucide-react';

interface SettingsPanelProps {
  settings: RequestSettings;
  onChange: (updated: RequestSettings) => void;
  errors: Record<string, string>;
  onReset: () => void;
  isOpenMobile?: boolean;
  onCloseMobile?: () => void;
}

export const SettingsPanel: React.FC<SettingsPanelProps> = ({
  settings,
  onChange,
  errors,
  onReset,
  isOpenMobile = false,
  onCloseMobile,
}) => {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const update = <K extends keyof RequestSettings>(key: K, value: RequestSettings[K]) => {
    onChange({
      ...settings,
      [key]: value,
    });
  };

  return (
    <aside className={`sidebar ${isOpenMobile ? 'open' : ''}`} aria-label="Workbench Settings">
      <div className="sidebar-header">
        <div className="brand-badge">
          <div className="brand-icon">
            <Zap size={18} />
          </div>
          <div>
            <div className="brand-title">AdaRoute</div>
            <div className="brand-subtitle">AI Gateway Workbench</div>
          </div>
        </div>
        {onCloseMobile && (
          <button
            type="button"
            className="mobile-menu-btn"
            onClick={onCloseMobile}
            aria-label="Close sidebar"
          >
            ✕
          </button>
        )}
      </div>

      <div className="sidebar-scroll">
        {/* Capability Selection */}
        <div className="control-group">
          <label className="control-label">Logical Capability</label>
          <div className="segmented-control" role="radiogroup" aria-label="Model Capability">
            {(['general', 'coding', 'reasoning'] as LogicalCapability[]).map((cap) => (
              <button
                key={cap}
                type="button"
                className={`segmented-btn ${settings.capability === cap ? 'active' : ''}`}
                onClick={() => update('capability', cap)}
                role="radio"
                aria-checked={settings.capability === cap}
              >
                {cap.charAt(0).toUpperCase() + cap.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {/* Privacy Level */}
        <div className="control-group">
          <label className="control-label">
            <span>Privacy Level</span>
            <Shield size={12} />
          </label>
          <select
            className="control-select"
            value={settings.privacyLevel}
            onChange={(e) => update('privacyLevel', e.target.value as PrivacyLevel)}
            aria-label="Privacy Level"
          >
            <option value="public">Public (Cloud Eligible)</option>
            <option value="internal_only">Internal Only (Local Fail-Closed)</option>
            <option value="confidential">Confidential (No Cache / Scrubbed)</option>
            <option value="restricted">Restricted (Strict Isolation)</option>
          </select>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)', marginTop: '0.15rem' }}>
            {settings.privacyLevel !== 'public'
              ? '🔒 Egress strictly forbidden. Local execution only.'
              : '🌐 Multi-provider eligible based on SLA and cost.'}
          </div>
        </div>

        {/* Budget */}
        <div className="control-group">
          <label className="control-label" htmlFor="setting-budget">
            <span>Budget Ceiling ($)</span>
          </label>
          <input
            id="setting-budget"
            type="number"
            step="0.001"
            min="0"
            className="control-input"
            value={settings.budget}
            onChange={(e) => update('budget', parseFloat(e.target.value) || 0)}
          />
          {errors.budget && <div className="control-error">{errors.budget}</div>}
        </div>

        {/* Quality Requirement */}
        <div className="control-group">
          <label className="control-label" htmlFor="setting-quality">
            <span>Quality Floor</span>
          </label>
          <select
            id="setting-quality"
            className="control-select"
            value={settings.qualityRequirement}
            onChange={(e) => update('qualityRequirement', e.target.value as QualityRequirement)}
          >
            <option value="low">Low (Fastest / Lowest Cost)</option>
            <option value="medium">Medium (Standard Balance)</option>
            <option value="high">High (Maximum Capability)</option>
          </select>
        </div>

        {/* Latency SLO Target */}
        <div className="control-group">
          <label className="control-label" htmlFor="setting-latency">
            <span>Latency Target (s)</span>
          </label>
          <input
            id="setting-latency"
            type="number"
            step="0.1"
            min="0.1"
            className="control-input"
            value={settings.latencySlo}
            onChange={(e) => update('latencySlo', parseFloat(e.target.value) || 0.1)}
          />
          {errors.latencySlo && <div className="control-error">{errors.latencySlo}</div>}
        </div>

        {/* Max Tokens */}
        <div className="control-group">
          <label className="control-label" htmlFor="setting-max-tokens">
            <span>Max Output Tokens</span>
          </label>
          <input
            id="setting-max-tokens"
            type="number"
            step="32"
            min="16"
            max="4096"
            className="control-input"
            value={settings.maxTokens}
            onChange={(e) => update('maxTokens', parseInt(e.target.value, 10) || 256)}
          />
          {errors.maxTokens && <div className="control-error">{errors.maxTokens}</div>}
        </div>

        {/* Advanced & Manual Testing Controls */}
        <div>
          <button
            type="button"
            className="disclosure-toggle"
            onClick={() => setShowAdvanced(!showAdvanced)}
            aria-expanded={showAdvanced}
          >
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Sliders size={14} />
              Advanced / Test Controls
            </span>
            {showAdvanced ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </button>

          {showAdvanced && (
            <div className="disclosure-content">
              {/* Routing Policy */}
              <div className="control-group">
                <label className="control-label" htmlFor="setting-policy">Routing Policy</label>
                <select
                  id="setting-policy"
                  className="control-select"
                  value={settings.policy}
                  onChange={(e) => update('policy', e.target.value as RoutingPolicy)}
                >
                  <option value="adaptive">Adaptive (Multi-Objective)</option>
                  <option value="strongest">Strongest (Max Quality)</option>
                  <option value="cheapest">Cheapest (Min Cost)</option>
                  <option value="static">Static (Task Matrix)</option>
                </select>
              </div>

              {/* Replica Scheduler */}
              <div className="control-group">
                <label className="control-label" htmlFor="setting-scheduler">Replica Scheduler</label>
                <select
                  id="setting-scheduler"
                  className="control-select"
                  value={settings.scheduler}
                  onChange={(e) => update('scheduler', e.target.value as ReplicaScheduler)}
                >
                  <option value="slo_aware">SLO-Aware (Queue Token Weighted)</option>
                  <option value="least_loaded">Least Loaded</option>
                  <option value="round_robin">Round Robin</option>
                </select>
              </div>

              {/* Cache Bypass */}
              <div className="control-group">
                <label className="checkbox-label" htmlFor="setting-no-cache">
                  <input
                    id="setting-no-cache"
                    type="checkbox"
                    checked={settings.noCache}
                    onChange={(e) => update('noCache', e.target.checked)}
                  />
                  <span>Bypass Cache (no_cache)</span>
                </label>
              </div>

              {/* Tenant ID */}
              <div className="control-group">
                <label className="control-label" htmlFor="setting-tenant">Tenant ID</label>
                <input
                  id="setting-tenant"
                  type="text"
                  className="control-input"
                  value={settings.tenantId}
                  onChange={(e) => update('tenantId', e.target.value)}
                  placeholder="default"
                />
              </div>

              {/* User ID */}
              <div className="control-group">
                <label className="control-label" htmlFor="setting-user">User ID</label>
                <input
                  id="setting-user"
                  type="text"
                  className="control-input"
                  value={settings.userId}
                  onChange={(e) => update('userId', e.target.value)}
                  placeholder="anonymous"
                />
              </div>

              {/* Gateway URL Override */}
              <div className="control-group">
                <label className="control-label" htmlFor="setting-gateway-url">Gateway URL Override</label>
                <input
                  id="setting-gateway-url"
                  type="text"
                  className="control-input"
                  value={settings.gatewayUrl}
                  onChange={(e) => update('gatewayUrl', e.target.value)}
                  placeholder="Direct: http://localhost:8000"
                />
                {errors.gatewayUrl && <div className="control-error">{errors.gatewayUrl}</div>}
              </div>
            </div>
          )}
        </div>

        {/* Reset Defaults */}
        <button
          type="button"
          className="action-btn"
          onClick={onReset}
          style={{ marginTop: 'auto', justifyContent: 'center', padding: '0.6rem' }}
        >
          <RotateCcw size={14} />
          Reset Controls to Defaults
        </button>
      </div>
    </aside>
  );
};
