import React, { useState } from 'react';
import { useChatState } from '../state/chatState';
import { SettingsPanel } from './SettingsPanel';
import { MessageList } from './MessageList';
import { Composer } from './Composer';
import { DiagnosticsPanel } from './DiagnosticsPanel';
import { ControlPlanePanel } from './ControlPlanePanel';
import { GatewayStatus } from './GatewayStatus';
import { Menu, Trash2, Zap, ShieldCheck } from 'lucide-react';

export const ChatShell: React.FC = () => {
  const {
    messages,
    inputPrompt,
    setInputPrompt,
    settings,
    updateSettings,
    resetSettings,
    validationErrors,
    isLoading,
    gatewayHealth,
    refreshHealth,
    sendMessage,
    cancelRequest,
    retryLast,
    clearConversation,
    activeDiagnostics,
    isDiagnosticsOpen,
    openDiagnostics,
    closeDiagnostics,
  } = useChatState();

  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);
  const [isControlPlaneOpen, setIsControlPlaneOpen] = useState(false);

  const handleClearWithConfirm = () => {
    if (messages.length === 0) return;
    if (window.confirm('Clear conversation history?')) {
      clearConversation();
    }
  };

  return (
    <div className="app-container">
      {/* Settings Sidebar */}
      <SettingsPanel
        settings={settings}
        onChange={updateSettings}
        errors={validationErrors}
        onReset={resetSettings}
        isOpenMobile={isMobileSidebarOpen}
        onCloseMobile={() => setIsMobileSidebarOpen(false)}
      />

      {/* Main Chat Area */}
      <main className="chat-main">
        {/* Chat Top Header */}
        <header className="chat-header">
          <div className="header-left">
            <button
              type="button"
              className="mobile-menu-btn"
              onClick={() => setIsMobileSidebarOpen(true)}
              aria-label="Open settings"
            >
              <Menu size={20} />
            </button>
            <div className="current-mode-indicator">
              <span>Capability:</span>
              <span className="mode-tag">{settings.capability}</span>
              <span style={{ color: 'var(--text-dim)' }}>•</span>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)', textTransform: 'capitalize' }}>
                {settings.privacyLevel.replace('_', ' ')}
              </span>
            </div>
          </div>

          <div className="header-right">
            <GatewayStatus health={gatewayHealth} onRefresh={refreshHealth} />

            <button
              type="button"
              className="action-btn"
              onClick={() => setIsControlPlaneOpen(true)}
              title="Open Auditable Control Plane & Cost Ledger"
              aria-label="Open Control Plane"
              style={{ color: 'var(--accent-primary)' }}
            >
              <ShieldCheck size={14} />
              <span>Control Plane</span>
            </button>

            {activeDiagnostics && (
              <button
                type="button"
                className="action-btn"
                onClick={() => openDiagnostics()}
                title="View latest routing telemetry"
                aria-label="View latest diagnostics"
                style={{ color: 'var(--accent-secondary)' }}
              >
                <Zap size={14} />
                <span>Diagnostics</span>
              </button>
            )}

            {messages.length > 0 && (
              <button
                type="button"
                className="action-btn"
                onClick={handleClearWithConfirm}
                title="Clear current conversation"
                aria-label="Clear conversation"
              >
                <Trash2 size={14} />
                <span>Clear</span>
              </button>
            )}
          </div>
        </header>

        {/* Message History & Suggestions */}
        <MessageList
          messages={messages}
          isLoading={isLoading}
          onRetry={retryLast}
          onSelectSuggestion={(prompt) => {
            setInputPrompt(prompt);
          }}
          onInspectDiagnostics={(diag) => openDiagnostics(diag)}
        />

        {/* Composer */}
        <Composer
          value={inputPrompt}
          onChange={setInputPrompt}
          onSend={() => sendMessage()}
          onCancel={cancelRequest}
          isLoading={isLoading}
          userPrivate={settings.userPrivate}
          onToggleUserPrivate={() => updateSettings({ ...settings, userPrivate: !settings.userPrivate })}
        />
      </main>

      {/* Diagnostics Slide-over Drawer */}
      <DiagnosticsPanel
        isOpen={isDiagnosticsOpen}
        onClose={closeDiagnostics}
        metadata={activeDiagnostics}
      />

      {/* Auditable Control Plane & Ledger */}
      <ControlPlanePanel
        isOpen={isControlPlaneOpen}
        onClose={() => setIsControlPlaneOpen(false)}
      />
    </div>
  );
};
