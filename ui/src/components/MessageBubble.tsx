import React, { useState } from 'react';
import { AppMessage, RoutingMetadata } from '../lib/types';
import { Copy, Check, RotateCcw, Zap, AlertTriangle, User, Bot } from 'lucide-react';

interface MessageBubbleProps {
  message: AppMessage;
  onRetry?: (content: string) => void;
  onInspectDiagnostics?: (diag: RoutingMetadata) => void;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({
  message,
  onRetry,
  onInspectDiagnostics,
}) => {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === 'user';
  const isError = Boolean(message.error);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard fallback
    }
  };

  const formatTime = (ts: number) => {
    return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className={`message-wrapper ${message.role}`}>
      <div className="message-meta">
        {isUser ? (
          <>
            <span>{formatTime(message.timestamp)}</span>
            <User size={12} />
            <span>You</span>
          </>
        ) : (
          <>
            <Bot size={12} />
            <span>{isError ? 'Gateway Notice' : 'AdaRoute Assistant'}</span>
            <span>{formatTime(message.timestamp)}</span>
          </>
        )}
      </div>

      <div className={`message-bubble ${message.role} ${isError ? 'error-bubble' : ''}`}>
        {isError && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.4rem', fontWeight: 600 }}>
            <AlertTriangle size={15} />
            <span>{message.error}</span>
          </div>
        )}
        <div>{message.content}</div>
      </div>

      <div className="message-actions">
        {!isUser && !isError && (
          <>
            <button
              type="button"
              className="action-btn"
              onClick={handleCopy}
              title="Copy message to clipboard"
              aria-label="Copy message"
            >
              {copied ? <Check size={12} color="var(--success)" /> : <Copy size={12} />}
              <span>{copied ? 'Copied' : 'Copy'}</span>
            </button>

            {message.diagnostics && onInspectDiagnostics && (
              <button
                type="button"
                className="action-btn"
                onClick={() => onInspectDiagnostics(message.diagnostics!)}
                title="Inspect routing diagnostics"
                aria-label="Inspect routing diagnostics"
              >
                <Zap size={12} color="var(--accent-secondary)" />
                <span>Diagnostics</span>
              </button>
            )}
          </>
        )}

        {isError && onRetry && (
          <button
            type="button"
            className="action-btn retry-btn"
            onClick={() => onRetry(message.content)}
            title="Retry failed request"
            aria-label="Retry request"
          >
            <RotateCcw size={12} />
            <span>Retry</span>
          </button>
        )}
      </div>
    </div>
  );
};
