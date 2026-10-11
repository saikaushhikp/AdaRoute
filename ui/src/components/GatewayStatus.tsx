import React from 'react';
import { GatewayHealth } from '../lib/types';
import { RefreshCw, Activity, AlertCircle, CheckCircle } from 'lucide-react';

interface GatewayStatusProps {
  health: GatewayHealth;
  onRefresh: () => void;
  isLoading?: boolean;
}

export const GatewayStatus: React.FC<GatewayStatusProps> = ({
  health,
  onRefresh,
  isLoading = false,
}) => {
  const getStatusLabel = () => {
    switch (health.status) {
      case 'healthy':
        return 'Gateway Active';
      case 'unavailable':
        return 'Gateway Offline';
      case 'checking':
        return 'Connecting...';
    }
  };

  const getStatusIcon = () => {
    if (isLoading || health.status === 'checking') {
      return <Activity size={12} className="animate-spin" />;
    }
    switch (health.status) {
      case 'healthy':
        return <CheckCircle size={12} />;
      case 'unavailable':
        return <AlertCircle size={12} />;
    }
  };

  const tooltipText =
    health.status === 'healthy'
      ? `Connected to AdaRoute Gateway\nActive Replicas: ${health.replicas?.length ?? 0}\nModels: ${health.models?.join(', ') || 'none'}`
      : health.error || 'Gateway is currently unreachable. Make sure the container or server is active.';

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
      <button
        className={`status-pill ${health.status}`}
        title={tooltipText}
        onClick={onRefresh}
        type="button"
        aria-label={`Gateway status: ${getStatusLabel()}`}
      >
        <span className="status-dot" />
        {getStatusIcon()}
        <span>{getStatusLabel()}</span>
      </button>
      <button
        type="button"
        className="action-btn"
        onClick={onRefresh}
        disabled={isLoading}
        title="Refresh gateway health"
        aria-label="Refresh gateway health"
      >
        <RefreshCw size={12} style={{ animation: isLoading ? 'spin 1s linear infinite' : 'none' }} />
      </button>
    </div>
  );
};
