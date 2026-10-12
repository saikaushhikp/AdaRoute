/**
 * AdaRoute Client Type Definitions
 * Represents the public API contract of AdaRoute Unified AI Gateway.
 */

export type LogicalCapability = 'general' | 'coding' | 'reasoning';

export type PrivacyLevel = 'public' | 'internal_only' | 'confidential' | 'restricted';

export type QualityRequirement = 'low' | 'medium' | 'high';

export type RoutingPolicy = 'adaptive' | 'strongest' | 'cheapest' | 'static';

export type ReplicaScheduler = 'slo_aware' | 'least_loaded' | 'round_robin';

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface RequestSettings {
  // End-user requirements
  capability: LogicalCapability;
  privacyLevel: PrivacyLevel;
  budget: number;
  qualityRequirement: QualityRequirement;
  latencySlo: number;
  maxTokens: number;
  userPrivate: boolean;

  // Advanced / manual testing overrides
  policy: RoutingPolicy;
  scheduler: ReplicaScheduler;
  tenantId: string;
  userId: string;
  noCache: boolean;
  gatewayUrl: string;
}

export interface RoutingMetadata {
  request_id: string;
  logical_model: string;
  selected_model: string;
  selected_replica: string;
  provider: string;
  latency_ms: number;
  attempts: number;
  cache_hit: boolean;
  estimated_cost: number;
  actual_cost: number;
  slo_attained: boolean;
  policy_name?: string;
}

export interface UsageInfo {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface ChatCompletionChoice {
  index?: number;
  message: ChatMessage;
  finish_reason?: string;
}

export interface ChatCompletionResponse {
  id: string;
  model: string;
  logical_model?: string;
  choices: ChatCompletionChoice[];
  usage?: UsageInfo;
  routing_metadata?: RoutingMetadata;
  content?: string;
  replica_id?: string;
  provider?: string;
}

export interface GatewayHealth {
  status: 'healthy' | 'unavailable' | 'checking';
  models?: string[];
  replicas?: string[];
  error?: string;
  checkedAt?: number;
}

export interface AppMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  error?: string;
  diagnostics?: RoutingMetadata;
}

export interface ClientError {
  status?: number;
  code: 'NETWORK_ERROR' | 'VALIDATION_ERROR' | 'ROUTING_REJECTED' | 'SERVER_ERROR' | 'TIMEOUT' | 'UNKNOWN';
  message: string;
  technicalDetail?: string;
}
