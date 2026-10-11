# AdaRoute Terminal Workbench CLI

An interactive, minimal, and high-performance terminal client for interacting with the AdaRoute AI Gateway. It replicates the functionality of the web UI directly in the console using rich formatting.

---

## 1. Quick Start

Run the CLI directly against a running AdaRoute Gateway:

```bash
# Direct port 8000 (standard gateway)
python3 ui/cli/main.py

# Or via reverse proxy / alternate URL
python3 ui/cli/main.py http://localhost:8080

# Or via justfile recipe
just cli
```

---

## 2. Interactive Commands

Within the workbench prompt, use slash commands to inspect or reconfigure gateway behavior:

| Command | Arguments | Description |
| :--- | :--- | :--- |
| `/help` | — | Display the command reference table. |
| `/settings` | — | Display active constraints, policies, and overrides. |
| `/diag` | — | View detailed routing telemetry for the last response. |
| `/health` | — | Query gateway cluster health and list active replicas. |
| `/clear` | — | Clear screen and reset conversation history. |
| `/exit` | — | Exit the CLI. |

### Configuration Commands (`/set`)

| Parameter | Options | Example |
| :--- | :--- | :--- |
| `/set cap <val>` | `general`, `coding`, `reasoning` | `/set cap coding` |
| `/set privacy <val>` | `public`, `internal_only`, `confidential`, `restricted` | `/set privacy internal_only` |
| `/set budget <float>` | Any non-negative dollar amount | `/set budget 0.005` |
| `/set quality <val>` | `low`, `medium`, `high` | `/set quality high` |
| `/set slo <seconds>` | Float > 0 | `/set slo 1.5` |
| `/set tokens <int>` | Integer > 0 | `/set tokens 512` |
| `/set policy <val>` | `adaptive`, `strongest`, `cheapest`, `static` | `/set policy strongest` |
| `/set scheduler <val>` | `slo_aware`, `least_loaded`, `round_robin` | `/set scheduler least_loaded` |
| `/set cache <on\|off>` | `on`, `off` | `/set cache off` |
| `/set url <url>` | Gateway endpoint | `/set url http://localhost:8080` |

---

## 3. Manual Testing Workflow via CLI

1. **Test Public Routing**:
   ```text
   user > Hello AdaRoute!
   ```
2. **Test Fail-Closed Privacy**:
   ```text
   user > /set privacy internal_only
   user > Analyze this internal database schema.
   user > /diag
   ```
   *(Verify that the selected model is strictly `qwen2.5:1.5b` and provider is `local`)*
3. **Test Cache Hit**:
   ```text
   user > Repeat exact prompt.
   user > Repeat exact prompt.
   user > /diag
   ```
   *(Verify `Cache Outcome: HIT` and `Estimated Cost: $0.000000`)*
4. **Test Budget Ceiling Rejection**:
   ```text
   user > /set budget 0.000001
   user > Run a high-cost query.
   ```
   *(Verify clean HTTP 503 error without stack traces)*
