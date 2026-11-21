# AdaRoute terminal workbench

The terminal workbench sends chat requests to a running AdaRoute gateway over HTTP and displays the response and routing metadata. It does not start the gateway or run a model itself.

## Start

From the repository root, install the Python dependencies and start the stack as described in the [main README](../../README.md):

```bash
python3 -m pip install -e '.[dev]'
just cli
```

The default gateway URL is `http://localhost:7000`. Pass a different base URL as the first argument, for example the web UI's reverse proxy:

```bash
python3 ui/cli/main.py http://localhost:7070
python3 ui/cli/main.py --help
```

The workbench checks `/health` when it starts. Type a prompt and press Enter to send it to `/v1/chat/completions`. `/diag` shows the routing metadata from the **last successful** response.

## Defaults

| Setting | Initial value |
|---|---|
| Capability | `general` |
| Privacy level | `public` |
| User private | Off |
| Conversation history | Off: each prompt is sent independently |
| Budget | `$1.0` per request |
| Quality requirement | `medium` |
| Latency SLO | `2.0` seconds |
| Maximum output tokens | `256` |
| Model policy | `adaptive` |
| Replica scheduler | `slo_aware` |
| Cache | Enabled (`no_cache=false`) |
| Tenant and user IDs | `user` / `user` |

The CLI does not expose commands for every API field: in particular, it does not set `execution_mode`, `temperature`, `session_id` or `session_budget`, and it does not send authentication headers. Requests use the gateway's configured execution mode. Use the HTTP API for those fields or for selecting a concrete model ID.

## Commands

| Command | Action |
|---|---|
| `/help` | Show the interactive command list. Aliases: `/h`, `/?`. |
| `/settings` | Show current settings. Alias: `/config`. |
| `/diag` | Show the last successful response's routing and attempt metadata. Alias: `/diagnostics`. |
| `/health` | Query the gateway and display models and active replicas. |
| `/history [on\|off]` | Toggle history, or explicitly enable/disable it. Alias: `/hist`. |
| `/private` | Toggle the local-only `user_private` setting. Alias: `/user_private`. |
| `/clear` | Clear the terminal and conversation history. Alias: `/cls`. |
| `/exit` | Exit. Aliases: `/quit`, `/q`; Ctrl+C and EOF also exit. |

Set request options with `/set <parameter> <value>`:

| Parameter | Values and examples |
|---|---|
| `cap` or `capability` | `general`, `coding`, `reasoning`; for example `/set cap coding` |
| `privacy` | `public`, `internal_only`, `confidential`, `restricted` |
| `budget` | Non-negative number in dollars; for example `/set budget 0.01` |
| `quality` | `low`, `medium`, `high` |
| `slo` | Positive number in seconds; for example `/set slo 3` |
| `tokens` | Integer maximum output length; for example `/set tokens 100` |
| `policy` | `adaptive`, `strongest`, `cheapest`, `static`, `vllm_semantic_router` |
| `scheduler` | `slo_aware`, `least_loaded`, `round_robin` |
| `history` or `hist` | `on` or `off`; disabling it clears retained history |
| `private` or `user_private` | `on` or `off` |
| `cache` | `on` or `off`; `off` sends `no_cache=true` |
| `url` | Gateway base URL; for example `/set url http://localhost:7070` |

The API supports additional configurable policies, but the CLI's `/set policy` accepts only the values listed above. `vllm_semantic_router` uses the configured preview client; the bundled service is an AdaRoute heuristic, not evidence that the upstream router is deployed. The CLI parses `/set tokens` as an integer; the gateway validates that it is positive.

## Common workflows

### Local-only request

```text
user > /set quality low
user > /private
user > Explain the difference between a process and a thread.
user > /diag
```

`/private` sets `user_private=true`; the gateway restricts eligible execution to local models. It may return an error if no eligible local model is healthy or satisfies the other constraints. `/private` toggles the setting, while `/set private on` and `/set private off` set it explicitly. The GPU profile's smaller models have configured quality scores below the default `medium` floor, so `low` is useful when testing that profile.

### Inspect cache behavior

```text
user > /history off
user > /set cache on
user > What is the chemical symbol for gold?
user > What is the chemical symbol for gold?
user > /diag
```

With identical compatible settings, the second request can return an exact hit. Existing cache entries, model availability, request eligibility and privacy settings affect the result. To force inference for an experiment, use `/set cache off`; use `/set cache on` to allow lookup again. The metadata may also identify a semantic hit when an eligible standalone prompt matches a stored entry.

History is off by default, so identical prompts are independent requests. With `/history on`, the CLI sends the accumulated user and assistant messages; changing the conversation changes the context used for caching. `/history off` and `/clear` clear the stored history. If a request fails while history is on, its user message has already been added to the local history; use `/clear` if you do not want it sent on the next turn.

### Compare local replica schedulers

```text
user > /set quality low
user > /set private on
user > /set cap coding
user > /set cache off
user > /set scheduler round_robin
user > Write a short Python comment.
user > Write a second short Python comment.
user > /diag
```

On the GPU profile, the Qwen2.5-Coder model has two separate replica endpoints. Inspect the selected model and replica in each response. Round robin cycles through healthy eligible replicas; an unavailable replica, a different selected model or a failed request can change the observed sequence. The CPU profile's replica labels do not by themselves establish separate engines.

`/health` reports the gateway's overall status and lists active replica IDs. The overall status can be healthy while an individual replica is unavailable; query `GET /health` directly for the complete per-replica details.

## Diagnostics and troubleshooting

- **Connection error:** confirm the gateway is running, check `/health`, and use `/set url` for the correct host and port. The HTTP client waits up to 60 seconds for a chat response.
- **No eligible deployment or HTTP 503:** inspect the requested privacy, quality floor, budget, execution mode and backend health. Private requests cannot fall back to hosted providers. GPU startup and model downloads can take several minutes.
- **Unexpected cache miss:** check that caching is on and the request, context and options are compatible. History mode changes the message list; confidential/restricted requests and `no_cache` bypass caching.
- **Unexpected simulated or unknown result:** inspect attempt metadata and the gateway's execution mode. For a strict real-only request, use the HTTP API with `execution_mode: "real"`.
- **Cost and quality:** displayed costs can be configured estimates or allocations rather than invoices. A model's configured quality floor is not a guarantee that its answer is correct.

The checked-in configuration is intended for a trusted local prototype. The CLI does not authenticate users, and the gateway's administrative endpoints should be secured before exposing the service to untrusted clients.
