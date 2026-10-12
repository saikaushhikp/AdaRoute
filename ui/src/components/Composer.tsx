import React, { useRef, useEffect } from 'react';
import { Send, StopCircle } from 'lucide-react';

interface ComposerProps {
  value: string;
  onChange: (val: string) => void;
  onSend: () => void;
  onCancel?: () => void;
  isLoading: boolean;
  disabled?: boolean;
  userPrivate?: boolean;
  onToggleUserPrivate?: () => void;
}

export const Composer: React.FC<ComposerProps> = ({
  value,
  onChange,
  onSend,
  onCancel,
  isLoading,
  disabled = false,
  userPrivate = false,
  onToggleUserPrivate,
}) => {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea height
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const scrollHeight = textareaRef.current.scrollHeight;
      textareaRef.current.style.height = `${Math.min(Math.max(scrollHeight, 48), 200)}px`;
    }
  }, [value]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isLoading && value.trim().length > 0 && !disabled) {
        onSend();
      }
    }
  };

  const isSendDisabled = isLoading || disabled || value.trim().length === 0;

  return (
    <div className="composer-area">
      <div className="composer-box">
        <textarea
          ref={textareaRef}
          className="composer-textarea"
          placeholder="Ask a question or provide an instruction (Shift+Enter for newline)..."
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled || isLoading}
          aria-label="Chat prompt input"
          rows={1}
        />
        <div className="composer-footer">
          <div className="composer-hints" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
            <span>Press <kbd style={{ background: 'var(--border-strong)', padding: '0.1rem 0.3rem', borderRadius: 3 }}>Enter</kbd> to send</span>
            {onToggleUserPrivate && (
              <button
                type="button"
                onClick={onToggleUserPrivate}
                title="When ON, strictly routes to local self-hosted models, excluding cloud models"
                style={{
                  background: userPrivate ? 'rgba(234, 179, 8, 0.2)' : 'transparent',
                  border: `1px solid ${userPrivate ? '#eab308' : 'var(--border-strong, #444)'}`,
                  color: userPrivate ? '#eab308' : 'var(--text-dim, #888)',
                  borderRadius: 4,
                  padding: '2px 8px',
                  cursor: 'pointer',
                  fontSize: '0.75rem',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 4,
                  fontWeight: userPrivate ? 600 : 400,
                }}
              >
                {userPrivate ? '🛡️ User Private: ON (Local)' : '☁️ User Private: OFF'}
              </button>
            )}
          </div>
          <div>
            {isLoading && onCancel ? (
              <button
                type="button"
                className="action-btn"
                onClick={onCancel}
                title="Cancel in-flight request"
                aria-label="Cancel request"
                style={{ color: 'var(--danger)' }}
              >
                <StopCircle size={16} />
                <span>Cancel</span>
              </button>
            ) : (
              <button
                type="button"
                className="send-btn"
                onClick={onSend}
                disabled={isSendDisabled}
                title="Send message to gateway"
                aria-label="Send message"
              >
                <span>Send</span>
                <Send size={14} />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
