# Problem Definition

## Measurable Routing Decision

The goal of the AdaRoute AI gateway is to choose a suitable model and replica for each request. The choice must follow rules about privacy, budget, and quality. For each request $r$ and current backend state $b$, we choose a routing action $a = (model, replica)$ that minimizes the total cost while satisfying these rules.

### Hard Constraints
- **Privacy**: $Privacy(a) = \text{allowed}$ (Sensitive requests must not be sent to providers that are not permitted to handle them).
- **Budget**: $Cost(a) \le B$ (The request must stay within its maximum allowed cost).
- **Required Quality**: $Quality(a) \ge Q_{\min}$ (The selected model must be good enough for the task).

### Soft Objectives / Preferences

Among the choices that satisfy the hard constraints, we minimize this combined cost function:
$J(a) = w_c C(a) + w_l L(a) + w_q Q_{loss}(a) + w_s SLO(a) + w_f F(a)$

Here:
- $C(a)$: Cost of executing action $a$
- $L(a)$: Expected response time, including time to the first token (TTFT) and time between tokens (TPOT)
- $Q_{loss}(a)$: Difference in quality compared with the strongest available model
- $SLO(a)$: Penalty when the request does not meet its latency target
- $F(a)$: Risk that the request fails or needs repeated attempts

## Scope Boundaries
- **Implementation**: We will build the gateway API, request and backend data models, routing policies, baseline routing methods, simulation runners, mock backends, and metrics collection.
- **Simulation**: When real multi-GPU hardware is unavailable, we will simulate external providers, queues for multiple replicas, GPU memory pressure, backend latency, and backend failures.
- **Hardware Assumptions**: The system must run on a single GPU, a CPU-only environment, or a simulation environment. Multi-GPU tests are optional and depend on available hardware.

# Assumptions

## Hardware and Resources
1. **Limited Resources**: Most testing will use one GPU, a CPU-only environment, or a simulation. Multi-GPU testing will be done only if the hardware is available.
2. **Mock Backends**: We will simulate external APIs and additional replicas. This lets us test routing under high load, growing queues, and failures without paying for cloud services or needing large hardware.

## Workloads
1. **Different Request Types**: Requests can have different prompt lengths, expected output lengths, task types, and latency requirements.
2. **Traffic Bursts**: Requests may arrive in short, heavy bursts. The system therefore needs reliable queue management and failover to another backend when needed.

## System State and Telemetry
1. **Delayed Telemetry**: The gateway will not always see the latest backend state. For example, queue size and memory pressure data may be slightly out of date.
2. **Missing Telemetry**: A backend may sometimes fail to report its state. When this happens, the gateway must use a safe routing policy that chooses the lowest-risk option.

# Literature and System Landscape

To place AdaRoute in context, we reviewed existing systems for serving and routing large language models. The table below shows what each system provides, what ideas we use, and what AdaRoute will add.

| Work / System | What it solves | Layer | What we reuse | What remains for AdaRoute |
| --- | --- | --- | --- | --- |
| **vLLM / PagedAttention** | Manages GPU memory used by cached model data | Model serving | A serving system to route requests to | We will not build a serving engine; we will route requests to vLLM |
| **RouteLLM** | Chooses models based on cost and quality | Model selection | Basic ideas for model routing | Combining model selection with current backend load |
| **SGLang** | Improves model execution by reusing cached data | Model serving | Ideas about caching and execution | A gateway policy for deciding when cached data can be shared between users |
| **Llumnix** | Moves work between model replicas as conditions change | Replica management | Ideas for scheduling replicas | Connecting gateway decisions to replicas without changing the serving engine |
| **DistServe** | Separates prompt processing from response generation | Model serving | Ideas about meeting response-time targets | We will measure these targets, but we will not build a split serving system |
| **Envoy** | Provides common gateway and failure-handling features | Infrastructure | Retry and circuit-breaker ideas | Applying these ideas to model-serving workloads |
| **Existing AI Gateways** (e.g. LiteLLM) | Provides one API with fallback and basic routing | Gateway | A reference for gateway structure | Routing across cost, quality, latency, and current backend load |

# Research Gap

Reliable LLM serving systems, such as vLLM and SGLang, and API gateways, such as LiteLLM and OpenRouter, already exist. However, there is still a missing layer that connects request requirements with the current state of the available backends.

**The Gap:**
Existing gateways mostly use fixed rules. For example, they may switch to another provider only after the first provider fails, or choose a provider using a fixed price list. Serving engines focus on using hardware efficiently, but they usually do not know about tenant budgets, privacy rules, or failover across different providers.

**Our Contribution:**
AdaRoute introduces a **Policy and Decision Layer** that chooses routes using three types of information at the same time:
1. **Request Characterization**: What does this request need?
2. **Current Backend State**: What are the queue and GPU conditions of each available replica?
3. **Policy Constraints**: What are the cost, privacy, and latency requirements?

The main idea is to separate **Model Selection** from **Replica Selection**. Model Selection chooses the type of model needed for the request. Replica Selection then chooses a specific running instance using its queue depth and hardware information. This separation makes it easier to balance cost, quality, and response time.

# System Architecture

AdaRoute is a small gateway layer that makes routing decisions before sending a request to a model-serving backend.

```text
                    CLIENT
                       |
                       v
              +-----------------+
              |   AdaRoute API  |
              +--------+--------+
                       |
                       v
             Request Characterizer
                       |
                       v
                 Policy Engine
                       |
          +------------+------------+
          |                         |
          v                         v
    Model/Provider              Cache Policy
       Router                       |
          |                         |
          +----------+--------------+
                     v
             Replica Scheduler
                     |
        +------------+------------+
        v            v            v
    Replica 1    Replica 2    External/Mock
        |            |            |
        +------------+------------+
                     v
                LLM Serving
                     |
                     v
         Metrics / Logs / Tracing
```

## Core Components
- **Request Characterizer**: Reads each API request and converts it into a structured `RequestContext`.
- **Policy Engine**: Checks the request rules and compares the available routing choices.
- **Model Router (Stage 1)**: Chooses the model capability needed for the request.
- **Replica Scheduler (Stage 2)**: Chooses a specific running instance of that model.
- **Mock Backends**: Simulate model providers and replicas when full GPU hardware is not available.

# Request Characterization

LLM requests can need different amounts of computation and service time. AdaRoute collects the main request details before choosing a route.

## Request Schema (`RequestContext`)
- **`task_type`**: The type of task, identified from request metadata or simple prompt rules. For example, a code block may indicate a code-related task.
- **`prompt_tokens`**: The number of tokens in the prompt, measured with a fast tokenizer such as tiktoken before routing.
- **`estimated_output_tokens`**: The expected response length, calculated by the Output-Length Estimator in document 07.
- **`quality_requirement`**: The quality level required by the application or tenant, such as `high`, `medium`, or `low`.
- **`latency_slo`**: The maximum response-time target set by the tenant, such as a two-second maximum time to the first token (TTFT).
- **`privacy_level`**: The data category, such as `internal_only` or `public`.
- **`budget`**: The maximum cost allowed for this request.
- **`tenant_id`**: The tenant identifier used to track quotas.

## Overhead
The characterization step must be fast. Its target overhead is less than 5 ms.

# Output-Length Estimation

Routing based on the *actual* output length would use information that is only known after generation. This would make the routing decision unrealistic and could expose information about the response. Therefore, we estimate the output length before inference.

## Estimator Function
$\hat{L}_{out} = f(\text{task}, \text{prompt\_length}, \text{max\_tokens}, \text{user\_history})$

For the initial W1 implementation, we use a simple baseline estimator:
1. If `max_tokens` is provided and is small, set the estimated output length to `max_tokens`.
2. Otherwise, use an average length for the task type. For example, we use 150 tokens for summarization, 300 tokens for code generation, and 50 tokens for question answering.

## Measurement of Estimator Error
During evaluation, we will compute the difference between the estimated and actual lengths:
$\text{Error} = |\hat{L}_{out} - L_{out\_actual}|$

We will then measure whether routing becomes less effective as this error increases.

# Routing Policy Specification (Model Selection)

The first stage chooses the model or provider that is suitable for the request. It selects the model type, not the specific running replica.

## Baselines
1. **StrongestModelPolicy**: Always chooses the most capable available model, such as a GPT-4-class model, without considering cost.
2. **CheapestModelPolicy**: Chooses the least expensive model that can meet the hard requirements.
3. **StaticRulePolicy**: Chooses a model from a fixed task rule, such as Code -> Model A or Chat -> Model B.

## Adaptive Routing Policy
The adaptive policy evaluates each eligible model using:
$a^* = \arg\min_a [ w_c C(a) + w_l L(a) + w_q Q_{loss}(a) ]$

The selected model must also satisfy:
- Privacy rules: internal data cannot be sent to external APIs.
- The request's budget limit.
- The minimum quality required for the task.

# Replica Scheduler Specification

Once the model type is chosen, the scheduler selects the specific backend replica that will handle the request.

## Replicas and Load

We measure the work waiting in each replica's queue instead of counting only the number of requests. A request with a long prompt or a long expected response needs more work than a short request.

$W_q = \sum_{i \in \text{queue}} (\text{prompt\_tokens}_i + \hat{\text{output\_tokens}}_i)$

## Baselines
1. **RoundRobinPolicy**: Sends requests to available replicas in order, such as $R_1 \rightarrow R_2 \rightarrow R_1$.
2. **LeastLoadedPolicy**: Sends each request to the replica with the fewest active requests.

## Adaptive / SLO-Aware Policy

The adaptive policy sends the request to the replica with the lowest $W_q$. It also applies penalties to replicas with recent latency spikes or high GPU memory use.

## Gradual Recovery

When a replica recovers, the gateway must not send all waiting traffic to it immediately. Instead, traffic will increase gradually, either by using a staged increase or by assigning requests with a controlled probability.

# Cache Safety Specification

Caching LLM responses can reduce cost and response time. However, unsafe caching could expose one tenant's data to another tenant or return an outdated response.

## Constraints
1. **Tenant Separation**: Cache keys must include `tenant_id` and `privacy_level` so that data is not shared across tenants or privacy levels.
2. **Expiration**: Cached semantic results must have a time-to-live (TTL) so that outdated facts are not served indefinitely.
3. **Bypass**: Requests marked with `no_cache` or a high privacy level must skip cache lookup and retrieval.

## Implementation Scope
For the initial prototype, we will implement exact-match caching with separate entries for each tenant. If time permits, we will study semantic and prefix caching while following the safety rules above.

# Resilience Specification

Failures are expected, so the gateway must detect them and respond in a controlled way.

## Telemetry
- **Missing Telemetry**: If a replica has not reported its health for $T$ seconds, mark it as `degraded`.
- **Unhealthy State**: If the number of consecutive failures reaches a set threshold, mark the replica as `offline`.

## Failover State Machine
1. **Primary Fails**: The request times out or the backend returns a 5xx error.
2. **Limited Retry**: Retry the request on another healthy replica of the *same* model class, with at most two retries.
3. **Fallback**: If all replicas of Model A fail, use Model B only if it still meets the request requirements. Otherwise, return an error immediately.
4. **No Silent Drops**: Every request that cannot be completed must have a log entry with the specific failure reason.

# Quality Evaluation

Quality should not be reduced to one overall number because different tasks need different measures. We will evaluate quality separately for each task category.

## Task Categories and Quality Metrics
1. **Factual QA**: Compare the answer with a reference answer using Exact Match (EM) or a meaning-based similarity score.
2. **Reasoning / Math**: Measure whether the final answer is correct.
3. **Coding**: Measure whether the first generated solution passes the unit tests (`pass@1`).
4. **Summarization**: Use ROUGE, BERTScore, or a language-model judge to check whether the summary is factually consistent.

## Reporting
We will report the metrics separately for each category. For example, an adaptive router may perform well on factual QA but poorly on coding. Separate results will make this difference visible.

# Baselines

We will compare different combinations of model selection and replica selection. These combinations provide simple reference points for evaluating the proposed policy.

| Model Policy | Replica Policy | Description |
| --- | --- | --- |
| Static | Round-robin | Uses one fixed model and sends requests to replicas in turn. |
| Static | Least-loaded | Uses one fixed model and sends each request to the replica with the fewest active requests. |
| Adaptive | Round-robin | Selects the model adaptively but sends requests to replicas in turn. |
| Adaptive | Least-loaded | Selects the model adaptively and uses the replica with the fewest active requests. |
| Adaptive | Adaptive (SLO-Aware) | Our proposed approach: selects both the model and replica using request needs, workload-weighted queues, and latency targets. |

# Experiment Matrix

To evaluate the two-stage policy layer, we will run the following scenarios.

## Scenarios
1. **Steady State**: Use a constant request rate with a mix of different task types.
2. **Bursty Workload**: Create sudden traffic spikes to test queue management and missed latency targets.
3. **Failover / Degraded Backend**: Take a replica offline or make it much slower to test failure handling.
4. **Budget Constrained**: Set strict budgets for different tenants to measure how the policy balances cost and quality.

We will run every scenario with all baseline combinations listed in `13_baselines.md`.

# Metrics Framework

The system will record and export the following measurements:

## Performance
- **TTFT**: Time to First Token, or the time until the first response token is returned
- **TPOT**: Time per Output Token, or the average time between response tokens
- **Throughput**: Number of requests and tokens processed per second
- **Tail Latency**: The 95th-percentile and 99th-percentile response times, which show the experience of the slowest requests

## Resources and Cost
- **GPU Utilization**: The percentage of GPU capacity used, either in simulation or on real hardware
- **Queue State**: The amount of work waiting over time, weighted by expected prompt and output length
- **Cost**: The estimated cost per successful request and per token

## Service Reliability
- **SLO Attainment**: The percentage of requests that meet their TTFT and TPOT targets
- **Failure Rate**: The percentage of requests that fail
- **Fallback Rate**: The percentage of requests sent to a secondary model

# Acceptance Tests

Before completing M2-M8, we will use automated policy-engine tests to verify these required behaviors:

1. **Safety**: A privacy-restricted request never selects an external backend that is not allowed to receive it.
2. **Budget**: If the request budget $B$ is lower than the cost of every allowed backend, the request is rejected safely.
3. **Reproducibility**: The same request, backend state, and configuration always produce the same routing decision.
4. **Missing Telemetry**: If backend state information is missing, the system uses a safe policy, such as the least-loaded healthy replica or round-robin selection.
5. **Replica Failure**: An unhealthy replica is removed from the available choices.
6. **Recovery**: A recovered replica returns to service gradually without receiving too much traffic at once.
7. **Traceability**: The system logs each request's characterization, model choice, replica choice, and retry attempts.

# Project Plan

## Progression Flow
`W0 Idea -> W1 Precise Specification -> M1-M7 Implementation -> M8 Experimental Evidence`

## Milestones
- **M1**: Stable AI gateway contract (API design and request schemas).
- **M2**: Multi-provider execution (Mock backends and local execution framework).
- **M3**: Adaptive Cost-Quality Router (Implementing the Policy Engine).
- **M4**: Safe Caching.
- **M5**: Resilience (Retries, failover).
- **M6**: SLO-aware Scheduler.
- **M7**: Observability (Metrics collection pipeline).
- **M8**: Reproducible Evaluation (Running the Experiment Matrix and publishing results).

This W1 specification acts as the contract for M1-M8 implementation.
