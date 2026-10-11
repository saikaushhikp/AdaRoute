import React, { useEffect, useRef } from 'react';
import { AppMessage, RoutingMetadata } from '../lib/types';
import { MessageBubble } from './MessageBubble';
import { Bot } from 'lucide-react';

interface MessageListProps {
  messages: AppMessage[];
  isLoading: boolean;
  onRetry: (promptText: string) => void;
  onSelectSuggestion: (promptText: string) => void;
  onInspectDiagnostics: (diag: RoutingMetadata) => void;
}

export const MessageList: React.FC<MessageListProps> = ({
  messages,
  isLoading,
  onRetry,
  onSelectSuggestion,
  onInspectDiagnostics,
}) => {
  const bottomRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom smoothly on new messages or loading state
  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages.length, isLoading]);

  const suggestions = [
    {
      title: 'Analyze Two-Stage Routing',
      prompt: 'Explain how AdaRoute separates Stage-1 model selection from Stage-2 replica scheduling.',
    },
    {
      title: 'Generate Benchmark Code',
      prompt: 'Write an asynchronous Python script using httpx to benchmark request latency and throughput.',
    },
    {
      title: 'Verify Privacy Isolation',
      prompt: 'What architectural guarantees prevent internal_only data from egressing to third-party providers?',
    },
    {
      title: 'Workload Queue Estimation',
      prompt: 'How does workload estimation W_q = sum(p_i + o_hat_i) prevent straggler replicas under concurrency?',
    },
  ];

  if (messages.length === 0) {
    return (
      <div className="messages-container" ref={containerRef}>
        <div className="empty-state">
          <div className="empty-state-icon">
            <Bot size={28} />
          </div>
          <h3>AdaRoute AI Gateway Workbench</h3>
          <p>
            An intelligent gateway combining zero-trust privacy fail-closed routing,
            dynamic queue tracking, composite caching, and multi-provider execution.
          </p>

          <div className="prompt-suggestions">
            {suggestions.map((s) => (
              <button
                key={s.title}
                type="button"
                className="suggestion-chip"
                onClick={() => onSelectSuggestion(s.prompt)}
              >
                <div style={{ fontWeight: 600, color: 'var(--text-main)', marginBottom: '0.2rem' }}>
                  {s.title}
                </div>
                <div>{s.prompt}</div>
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="messages-container" ref={containerRef}>
      {messages.map((msg) => (
        <MessageBubble
          key={msg.id}
          message={msg}
          onRetry={onRetry}
          onInspectDiagnostics={onInspectDiagnostics}
        />
      ))}

      {isLoading && (
        <div className="message-wrapper assistant">
          <div className="message-meta">
            <Bot size={12} />
            <span>AdaRoute Gateway (Routing...)</span>
          </div>
          <div className="message-bubble assistant">
            <div className="typing-dots">
              <span className="typing-dot" />
              <span className="typing-dot" />
              <span className="typing-dot" />
            </div>
          </div>
        </div>
      )}

      <div ref={bottomRef} style={{ height: 1 }} />
    </div>
  );
};
