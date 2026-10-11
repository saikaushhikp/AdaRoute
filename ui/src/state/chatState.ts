import { useState, useEffect, useRef, useCallback } from 'react';
import {
  AppMessage,
  ChatMessage,
  ClientError,
  GatewayHealth,
  RequestSettings,
  RoutingMetadata,
} from '../lib/types';
import { GatewayClient } from '../lib/gatewayClient';
import { validatePrompt, validateSettings } from '../lib/validation';

export const DEFAULT_SETTINGS: RequestSettings = {
  capability: 'general',
  privacyLevel: 'public',
  budget: 1.0,
  qualityRequirement: 'medium',
  latencySlo: 2.0,
  maxTokens: 256,
  policy: 'adaptive',
  scheduler: 'slo_aware',
  tenantId: 'default',
  userId: 'anonymous',
  noCache: false,
  gatewayUrl: '',
};

export function useChatState(clientInstance?: GatewayClient) {
  const [client, setClient] = useState<GatewayClient>(() => clientInstance || new GatewayClient());
  const [messages, setMessages] = useState<AppMessage[]>([]);
  const [inputPrompt, setInputPrompt] = useState('');
  const [settings, setSettings] = useState<RequestSettings>(DEFAULT_SETTINGS);
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({});
  const [isLoading, setIsLoading] = useState(false);
  const [activeDiagnostics, setActiveDiagnostics] = useState<RoutingMetadata | null>(null);
  const [isDiagnosticsOpen, setIsDiagnosticsOpen] = useState(false);
  const [gatewayHealth, setGatewayHealth] = useState<GatewayHealth>({
    status: 'checking',
  });

  const abortControllerRef = useRef<AbortController | null>(null);

  // Update client base URL when settings.gatewayUrl changes
  useEffect(() => {
    client.setBaseUrl(settings.gatewayUrl);
  }, [client, settings.gatewayUrl]);

  // Initial and periodic health check
  const refreshHealth = useCallback(async () => {
    try {
      const health = await client.checkHealth();
      setGatewayHealth(health);
    } catch {
      setGatewayHealth({ status: 'unavailable', error: 'Could not connect' });
    }
  }, [client]);

  useEffect(() => {
    refreshHealth();
    // Modest background refresh every 30 seconds
    const interval = setInterval(refreshHealth, 30000);
    return () => clearInterval(interval);
  }, [refreshHealth]);

  const updateSettings = (newSettings: RequestSettings) => {
    setSettings(newSettings);
    const { errors } = validateSettings(newSettings);
    setValidationErrors(errors);
  };

  const resetSettings = () => {
    setSettings(DEFAULT_SETTINGS);
    setValidationErrors({});
    client.setBaseUrl('');
  };

  const cancelRequest = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsLoading(false);
  };

  const sendMessage = async (promptOverride?: string) => {
    const rawPrompt = promptOverride !== undefined ? promptOverride : inputPrompt;
    const promptError = validatePrompt(rawPrompt);
    if (promptError) {
      return;
    }

    const { isValid, errors } = validateSettings(settings);
    if (!isValid) {
      setValidationErrors(errors);
      return;
    }

    // Build user message
    const userMessage: AppMessage = {
      id: `usr-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
      role: 'user',
      content: rawPrompt.trim(),
      timestamp: Date.now(),
    };

    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    if (promptOverride === undefined) {
      setInputPrompt('');
    }

    setIsLoading(true);
    abortControllerRef.current = new AbortController();

    // Prepare message history for OpenAI-style messages list
    const apiMessages: ChatMessage[] = newMessages
      .filter((m) => !m.error)
      .map((m) => ({
        role: m.role,
        content: m.content,
      }));

    try {
      const response = await client.sendChatCompletion(
        apiMessages,
        settings,
        abortControllerRef.current.signal
      );

      const content =
        response.content ||
        (response.choices && response.choices[0]?.message?.content) ||
        'No response content received.';

      const assistantMessage: AppMessage = {
        id: response.id || `ast-${Date.now()}`,
        role: 'assistant',
        content,
        timestamp: Date.now(),
        diagnostics: response.routing_metadata,
      };

      setMessages((prev) => [...prev, assistantMessage]);
      if (response.routing_metadata) {
        setActiveDiagnostics(response.routing_metadata);
      }
    } catch (err: unknown) {
      const clientErr = err as ClientError;
      const errorMessage: AppMessage = {
        id: `err-${Date.now()}`,
        role: 'assistant',
        content: rawPrompt.trim(), // Preserve original input for easy retry
        timestamp: Date.now(),
        error: clientErr.message || 'An unexpected error occurred.',
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  };

  const retryLast = (content: string) => {
    sendMessage(content);
  };

  const clearConversation = () => {
    cancelRequest();
    setMessages([]);
    setActiveDiagnostics(null);
  };

  const openDiagnostics = (diag?: RoutingMetadata) => {
    if (diag) {
      setActiveDiagnostics(diag);
    }
    setIsDiagnosticsOpen(true);
  };

  const closeDiagnostics = () => {
    setIsDiagnosticsOpen(false);
  };

  return {
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
    setClient,
  };
}
