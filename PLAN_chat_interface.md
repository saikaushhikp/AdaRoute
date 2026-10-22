# Plan: AdaRoute Chat Interface

## 1. Purpose

Build a small, polished, end-user chat interface in `/home/kaushik/AdaRoute/ui/` that exercises AdaRoute through its public HTTP API.

The interface is a separate client application, not part of the gateway implementation. A user should be able to chat without knowing which model, provider, replica, cache, retry policy, or routing decision handled the request.

The interface must also support a deliberate manual-testing workflow so a developer can impersonate an end user and observe whether privacy, budget, model-selection, caching, scheduling, resilience, and error behavior work correctly.

## 2. Current Gateway Contract

The client must integrate with the existing gateway contract rather than importing Python modules from `adaroute`.

### Gateway endpoints

- `GET /health`
- `POST /v1/chat/completions`

The current completion endpoint accepts a JSON request containing fields such as:

```json
{
  "model": "general",
  "prompt": "Hello",
  "privacy_level": "public",
  "budget": 1.0,
  "quality_requirement": "medium",
  "latency_slo": 2.0,
  "tenant_id": "default",
  "user_id": "anonymous",
  "policy": "adaptive",
  "scheduler": "slo_aware",
  "no_cache": false,
  "max_tokens": 256
}
```

The endpoint also accepts a standard `messages` list. The UI should send `messages` for normal chat interactions and preserve the gateway's logical model aliases: `general`, `coding`, and `reasoning`.

Successful responses contain:

- assistant content
- usage information
- request ID
- selected model
- selected replica
- provider
- latency
- attempts
- cache-hit state
- estimated and actual cost
- SLO attainment
- policy name

The UI must treat the response as an external API response and handle missing, malformed, or unexpected fields defensively.

## 3. Recommended Technology

Use a separate frontend application with:

- React
- TypeScript
- Vite
- modern CSS or a small CSS utility layer
- browser `fetch` or a small typed HTTP client
- Docker with a multi-stage build
- Nginx or an equivalent static production server

Do not add a second Python web server for the UI. Do not import `adaroute` source code into the UI. The boundary between the UI and gateway must remain HTTP.

The implementation may choose another stack if it produces a simpler, accessible, maintainable result, but it must preserve the same container and HTTP boundary.

## 4. Product and UI Direction

The first screen should be the usable chat experience, not a marketing page.

The visual language should feel like a focused AI workbench:

- calm, high-contrast, readable interface
- purposeful typography rather than default browser styling
- clear message hierarchy between user and assistant
- restrained visual decoration
- responsive layout for desktop and mobile
- compact controls that do not overwhelm the conversation
- no provider or infrastructure terminology in the primary user flow

The interface must not expose technical routing metadata by default. Infrastructure details belong in an optional diagnostics surface.

## 5. Required User Features

### 5.1 Chat interaction

Implement:

- conversation history containing user and assistant messages
- multiline prompt input
- submit button with a familiar send icon
- `Enter` to send and `Shift+Enter` for a newline
- disabled submit state while a request is pending
- clear conversation action with confirmation or an undo-friendly interaction
- empty, loading, success, and error states
- automatic scroll to the newest message without preventing manual scrolling
- retry action for the most recent failed request
- copy assistant response action
- accessible labels and keyboard navigation

The interface should preserve the visible conversation during a failed request. It must not silently discard user input.

### 5.2 User-facing request controls

Provide a compact settings area for controls that represent realistic end-user requirements:

- logical capability: General, Coding, Reasoning
- privacy level: Public, Internal only, Confidential, Restricted
- budget
- quality requirement
- latency target/SLO
- maximum output tokens

Use appropriate controls:

- select or segmented control for logical capability
- select for privacy and quality
- numeric input with validation for budget and latency
- numeric input or stepper for maximum tokens

Defaults should be conservative and match the gateway defaults where practical. The UI must explain validation failures in plain language, without requiring the user to understand routing internals.

### 5.3 Optional advanced/manual-testing controls

Place advanced controls behind a clearly labeled `Advanced` or `Test controls` disclosure. These controls are for manual verification, not the default user experience:

- routing policy: Adaptive, Strongest, Cheapest, Static
- replica scheduler: SLO-aware, Least loaded, Round robin
- tenant ID
- user ID
- bypass cache (`no_cache`)
- gateway base URL display or override when running outside Docker

Advanced values must be sent only when explicitly selected. Avoid accidentally making every normal user request look like a synthetic test request.

### 5.4 Diagnostics panel

After a successful request, allow the tester to open a diagnostics drawer or panel showing:

- request ID
- selected logical and concrete model
- provider
- replica
- latency
- attempts
- cache hit/miss
- estimated cost
- actual cost
- SLO attained/not attained
- policy used

Diagnostics must be visually secondary and must not expose raw prompts, hidden secrets, or internal stack traces.

### 5.5 Gateway status

Add a small status indicator that checks `GET /health`:

- healthy
- unavailable
- checking

Show a concise error and a retry action when the gateway cannot be reached. Do not poll aggressively; use an initial check and a modest refresh strategy.

## 6. Request and Response Behavior

### Request construction

For a normal chat submission:

1. Validate the form locally.
2. Add the user message to the visible conversation.
3. Send the conversation as `messages` to `POST /v1/chat/completions`.
4. Include the selected request constraints.
5. Show a pending assistant state.
6. Parse the response.
7. Add the assistant content to the conversation.
8. Store the latest response metadata for diagnostics.

Do not send a request if:

- the prompt is empty or whitespace-only
- budget is negative or not numeric
- latency target is invalid
- maximum tokens is invalid
- required gateway settings are missing

### Error mapping

Handle at least:

- network failure or gateway unavailable
- HTTP 400 validation failure
- HTTP 422 malformed request
- HTTP 503 routing rejection, unavailable replicas, or inference failure
- unexpected response shape
- request timeout or cancellation

Show user-friendly messages. Preserve a developer-visible technical detail only in diagnostics or a development console, and never display raw stack traces in the main chat.

The current gateway is non-streaming. Implement a clean request/response experience first. Keep the client API layer isolated so streaming can be added later without rewriting the conversation components.

## 7. Application Structure

Create the UI under `/home/kaushik/AdaRoute/ui/` with a maintainable structure similar to:

```text
ui/
  package.json
  package-lock.json or pnpm-lock.yaml
  tsconfig.json
  vite.config.ts
  index.html
  Dockerfile
  nginx.conf
  .dockerignore
  src/
    main.tsx
    App.tsx
    styles/
    components/
      ChatShell.tsx
      MessageList.tsx
      MessageBubble.tsx
      Composer.tsx
      SettingsPanel.tsx
      DiagnosticsPanel.tsx
      GatewayStatus.tsx
    lib/
      gatewayClient.ts
      types.ts
      validation.ts
    state/
      chatState.ts
```

The exact structure may vary, but responsibilities must remain separated:

- presentation components render UI
- gateway client owns HTTP calls and response parsing
- types describe request/response contracts
- validation owns client-side validation
- state owns conversation and request lifecycle

Avoid putting fetch calls, routing metadata parsing, and complex state transitions directly in one large component.

## 8. Container and Networking Design

The UI must run in its own container and communicate with AdaRoute over HTTP/TCP through a Docker network.

### Development and deployment expectations

- Gateway container exposes its FastAPI port, currently expected to be `8000`.
- UI container serves the built frontend on its own port, for example `8080`.
- A Docker Compose file may be added at repository root to run both services together.
- The UI must use a configurable gateway base URL; do not hard-code `localhost` into production code.
- In a Compose network, the gateway should be addressed by its service name, such as `http://gateway:8000`.
- When the browser itself must call the gateway, configure a reverse proxy or an explicit CORS policy. Prefer a reverse proxy so the browser can call the UI origin without exposing environment-specific cross-origin details.

Recommended production path:

```text
Browser -> UI container/Nginx -> gateway service over Docker network -> AdaRoute
```

The UI container must not need access to the gateway's source filesystem or Python environment.

Document:

- how to build the UI image
- how to start the gateway and UI together
- the URL to open in a browser
- how to override the gateway URL for local development
- how to run the UI against a separately hosted gateway

## 9. Security and Privacy Requirements

- Never log prompt content by default.
- Never place gateway credentials or provider credentials in browser code.
- Do not add API keys to frontend source or Docker image layers.
- Do not render raw backend exception messages to normal users.
- Keep tenant/user fields clearly marked as test/demo fields if exposed.
- Treat the UI as an untrusted client: all privacy and budget guarantees must remain enforced by the gateway.
- Do not claim that a client-side privacy selector provides security; it only declares the request requirement to the gateway.
- Avoid storing sensitive conversations in local storage by default.
- If browser persistence is later added, make it opt-in and document its privacy implications.

## 10. Manual End-User Test Scenarios

The completed UI must make these scenarios easy to perform manually:

1. Public general chat succeeds.
2. Coding and reasoning capability selections reach the gateway.
3. Internal-only, confidential, and restricted requests do not route to an external provider.
4. An impossible budget produces a clear rejection without displaying a raw stack trace.
5. A negative budget is rejected before submission or returns a clear gateway validation error.
6. Repeating an identical eligible request demonstrates a cache hit in diagnostics.
7. Enabling `no_cache` demonstrates a cache bypass.
8. Selecting different policies changes the visible diagnostic routing result where the backend state permits it.
9. Selecting different schedulers is reflected in the request and diagnostics.
10. A gateway outage produces a recoverable offline state and retry action.
11. A failed request preserves the user's message and allows retry.
12. A long prompt and a short prompt remain usable on desktop and mobile.
13. Keyboard-only navigation can complete a request.
14. Mobile viewport does not clip the composer, settings, messages, or diagnostics.
15. The UI does not reveal raw prompt text in browser logs or diagnostics metadata.

Manual testing should verify both visible behavior and the corresponding gateway response metadata.

## 11. Automated UI Validation

Add focused tests for the UI where the selected stack supports them:

- component tests for message rendering, validation, loading, and error states
- gateway client tests for successful responses, HTTP errors, malformed responses, and network errors
- an end-to-end smoke test covering opening the app, checking gateway status, sending a message, and viewing diagnostics
- responsive/accessibility checks for keyboard operation and important labels

Do not replace real behavior with mock-only assertions. Gateway-client tests may use a controlled HTTP test server, while at least one end-to-end path should run against a real gateway container.

## 12. Implementation Sequence

### Phase 1: Scaffold

- Create the UI project under `/home/kaushik/AdaRoute/ui/`.
- Add the selected frontend toolchain and scripts.
- Establish formatting, linting, and type checking.
- Add a minimal Docker build and local development command.

### Phase 2: Gateway client

- Define TypeScript request and response types.
- Implement configurable base URL handling.
- Implement health check.
- Implement completion request with timeout and cancellation support.
- Add defensive response parsing and typed error mapping.

### Phase 3: Core chat experience

- Build responsive chat shell.
- Add conversation state and request lifecycle.
- Add composer, loading state, error state, retry, clear, and copy actions.
- Ensure normal mode hides infrastructure details.

### Phase 4: Settings and diagnostics

- Add user-facing request controls.
- Add advanced/manual-testing controls behind disclosure.
- Add diagnostics drawer using the returned routing metadata.
- Add gateway health indicator.

### Phase 5: Container integration

- Add production Dockerfile.
- Add Nginx/reverse-proxy configuration if needed.
- Add Docker Compose integration for UI plus gateway.
- Verify service-name networking and browser access.

### Phase 6: Validation and documentation

- Run type checking, linting, unit/component tests, and UI smoke tests.
- Execute the manual test matrix above.
- Document startup commands, URLs, environment variables, and known limitations.
- Confirm the UI works with the current non-streaming gateway.

## 13. Definition of Done

The work is complete when:

- `/home/kaushik/AdaRoute/ui/` contains a working chat interface.
- The UI runs independently from the gateway in its own container.
- The UI communicates with AdaRoute only through HTTP.
- A normal user can send and receive chat messages without seeing routing complexity.
- A tester can configure privacy, budget, quality, latency, model alias, policy, scheduler, and cache behavior.
- Gateway health and failures are understandable from the UI.
- Routing metadata is available through an optional diagnostics view.
- No secrets or raw prompts are logged or exposed unnecessarily.
- Docker networking works using service names rather than host-specific assumptions.
- The documented manual scenarios pass.
- Automated checks pass, and the final verification includes a real UI-to-gateway request.

## 14. Known Scope Limits

- Do not implement model inference in the UI.
- Do not move routing, privacy enforcement, budget enforcement, caching, retry, or scheduling logic into the UI.
- Do not add streaming unless the gateway contract is extended and tested for streaming.
- Do not add authentication or multi-user identity claims unless the gateway contract supports them; tenant and user fields are currently request metadata, not an authentication system.
- Do not turn the first version into a general-purpose production chat platform. Keep it a reliable, comfortable client and manual-testing surface for AdaRoute.
