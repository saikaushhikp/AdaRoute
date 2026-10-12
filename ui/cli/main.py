#!/usr/bin/env python3
"""AdaRoute Terminal Workbench CLI.
A minimal, fast, and good-looking terminal interface for interacting with
the AdaRoute AI Gateway and testing routing policies, replica schedulers,
privacy constraints, and caching.
"""

import sys
from typing import Any

from client import GatewayCLIClient
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


class AdaRouteCLI:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.client = GatewayCLIClient(base_url)
        self.messages: list[dict[str, str]] = []
        self.last_metadata: dict[str, Any] | None = None

        # Request configuration
        self.config = {
            "capability": "general",
            "privacy_level": "public",
            "budget": 1.0,
            "quality_requirement": "medium",
            "latency_slo": 2.0,
            "max_tokens": 256,
            "policy": "adaptive",
            "scheduler": "slo_aware",
            "tenant_id": "user",
            "user_id": "user",
            "no_cache": False,
            "user_private": False,
            "history": False,  # Default: False (single-query workbench testing mode)
        }

    def print_header(self) -> None:
        """Prints the workbench header and live health status."""
        health = self.client.check_health()
        is_healthy = health.get("status") == "healthy"
        status_text = Text("● Online", style="bold green") if is_healthy else Text("● Offline", style="bold red")

        header_table = Table.grid(padding=(0, 2))
        header_table.add_column()
        header_table.add_column()

        privacy_color = "green" if self.config["privacy_level"] == "public" else "yellow"
        if self.config["privacy_level"] in ("confidential", "restricted"):
            privacy_color = "red"

        priv_status = "[bold yellow]ON (local only)[/bold yellow]" if self.config["user_private"] else "[dim]OFF[/dim]"
        hist_status = "[bold green]ON[/bold green]" if self.config["history"] else "[dim]OFF (single-query)[/dim]"

        header_table.add_row(
            Text.from_markup(f"[bold blue]⚡ AdaRoute AI Gateway[/bold blue] [dim]({self.client.base_url})[/dim]"),
            status_text,
        )
        header_table.add_row(
            Text.from_markup(
                f"[dim]Cap:[/dim] [cyan]{self.config['capability']}[/cyan]  "
                f"[dim]Privacy:[/dim] [{privacy_color}]{self.config['privacy_level']}[/{privacy_color}]  "
                f"[dim]User Private:[/dim] {priv_status}  "
                f"[dim]History:[/dim] {hist_status}  "
                f"[dim]Budget:[/dim] ${self.config['budget']:.4f}  "
                f"[dim]Policy:[/dim] {self.config['policy']}"
            ),
            Text.from_markup("[dim]Type [bold white]/help[/bold white] for commands[/dim]"),
        )

        panel = Panel(
            header_table,
            border_style="blue",
            box=box.ROUNDED,
            title="[bold white]AdaRoute Workbench[/bold white]",
            subtitle="[dim]Press Ctrl+C or /exit to quit[/dim]",
        )
        console.print(panel)

    def print_help(self) -> None:
        """Prints help and available commands."""
        table = Table(title="Available CLI Commands", box=box.SIMPLE_HEAVY)
        table.add_column("Command", style="cyan", no_wrap=True)
        table.add_column("Arguments", style="yellow")
        table.add_column("Description", style="white")

        table.add_row("/help", "", "Show this help table")
        table.add_row("/settings", "", "Display all current routing & request constraints")
        table.add_row("/set cap", "<general|coding|reasoning>", "Set target logical capability")
        table.add_row(
            "/set privacy",
            "<public|internal_only|confidential|restricted>",
            "Set privacy isolation level",
        )
        table.add_row("/set budget", "<float>", "Set max request budget in USD")
        table.add_row("/set quality", "<low|medium|high>", "Set minimum quality floor")
        table.add_row("/set slo", "<seconds>", "Set target latency SLO")
        table.add_row("/set tokens", "<int>", "Set maximum output tokens")
        table.add_row(
            "/set policy",
            "<adaptive|strongest|cheapest|static>",
            "Set Stage 1 model policy",
        )
        table.add_row(
            "/set scheduler",
            "<slo_aware|least_loaded|round_robin>",
            "Set Stage 2 replica scheduler",
        )
        table.add_row("/history", "[on|off]", "Toggle conversation history (default: OFF for caching tests)")
        table.add_row("/set history", "<on|off>", "Enable/disable multi-turn context retention")
        table.add_row("/set cache", "<on|off>", "Toggle caching (off sets no_cache=True)")
        table.add_row("/set url", "<url>", "Change gateway base URL")
        table.add_row("/diag", "", "Inspect detailed routing telemetry for the last response")
        table.add_row("/health", "", "Check gateway cluster health & active replicas")
        table.add_row("/clear", "", "Clear conversation history & reset screen")
        table.add_row("/exit", "", "Quit the workbench")

        console.print(table)

    def print_settings(self) -> None:
        """Displays formatted settings table."""
        table = Table(title="Active Request Settings", box=box.ROUNDED)
        table.add_column("Setting", style="bold cyan")
        table.add_column("Value", style="bold white")
        table.add_column("Category", style="dim")

        table.add_row("Logical Capability", self.config["capability"], "User Requirement")
        table.add_row("Privacy Level", self.config["privacy_level"], "User Requirement")
        table.add_row("Conversation History", str(self.config["history"]), "Workbench Mode")
        table.add_row("Budget Ceiling", f"${self.config['budget']:.4f}", "Economic Constraint")
        table.add_row("Quality Floor", self.config["quality_requirement"], "SLA Constraint")
        table.add_row("Latency SLO", f"{self.config['latency_slo']}s", "SLA Constraint")
        table.add_row("Max Output Tokens", str(self.config["max_tokens"]), "Model Parameter")
        table.add_row("Routing Policy", self.config["policy"], "Stage 1 Policy")
        table.add_row("Replica Scheduler", self.config["scheduler"], "Stage 2 Scheduler")
        table.add_row("Cache Bypass (no_cache)", str(self.config["no_cache"]), "Test Control")
        table.add_row("Gateway Base URL", self.client.base_url, "Infrastructure")

        console.print(table)

    def print_diagnostics(self) -> None:
        """Displays full telemetry from the latest request."""
        if not self.last_metadata:
            console.print("[yellow]No routing telemetry available yet. Send a request first.[/yellow]")
            return

        meta = self.last_metadata
        table = Table(title="Last Routing Telemetry", box=box.ROUNDED)
        table.add_column("Metric / Field", style="bold cyan")
        table.add_column("Value", style="white")

        table.add_row("Request ID", meta.get("request_id", "N/A"))
        table.add_row("Requested Capability", meta.get("logical_model", "N/A"))
        table.add_row("Selected Concrete Model", f"[bold green]{meta.get('selected_model', 'N/A')}[/bold green]")
        table.add_row("Selected Serving Replica", f"[bold yellow]{meta.get('selected_replica', 'N/A')}[/bold yellow]")
        table.add_row("Execution Provider", meta.get("provider", "N/A"))
        table.add_row("Policy Applied", meta.get("policy_name", "adaptive"))
        table.add_row("Latency (ms)", f"{meta.get('latency_ms', 0):.2f} ms")
        table.add_row("Attempts", str(meta.get("attempts", 1)))

        cache_hit = meta.get("cache_hit", False)
        cache_badge = "[bold green]HIT[/bold green]" if cache_hit else "[dim]MISS[/dim]"
        table.add_row("Cache Outcome", cache_badge)

        table.add_row("Estimated Cost", f"${meta.get('estimated_cost', 0.0):.6f}")
        table.add_row("Actual Cost", f"${meta.get('actual_cost', 0.0):.6f}")

        slo_met = meta.get("slo_attained", True)
        slo_badge = "[green]Attained[/green]" if slo_met else "[red]Breached[/red]"
        table.add_row("SLO Attainment", slo_badge)

        console.print(table)

    def check_and_print_health(self) -> None:
        """Queries and displays live cluster health."""
        health = self.client.check_health()
        if health.get("status") == "healthy":
            console.print("[bold green]✔ Gateway is healthy and reachable![/bold green]")
            console.print(f"[dim]Available Models:[/dim] {', '.join(health.get('models', []))}")
            console.print(f"[dim]Active Replicas:[/dim] {', '.join(health.get('replicas', []))}")
        else:
            console.print(f"[bold red]✖ Gateway is unavailable:[/bold red] {health.get('error', 'Unknown')}")

    def handle_set(self, args: list[str]) -> None:
        """Parses and updates settings."""
        if len(args) < 2:
            console.print("[red]Usage: /set <param> <value>[/red]")
            return

        param, val = args[0].lower(), args[1].lower()

        if param in ("cap", "capability"):
            if val in ("general", "coding", "reasoning"):
                self.config["capability"] = val
                console.print(f"[green]Capability set to '{val}'[/green]")
            else:
                console.print("[red]Invalid capability. Choose: general, coding, reasoning[/red]")

        elif param == "privacy":
            if val in ("public", "internal_only", "confidential", "restricted"):
                self.config["privacy_level"] = val
                console.print(f"[green]Privacy level set to '{val}'[/green]")
            else:
                console.print("[red]Invalid privacy. Choose: public, internal_only, confidential, restricted[/red]")

        elif param == "budget":
            try:
                b = float(val)
                if b < 0:
                    console.print("[red]Budget must be non-negative.[/red]")
                else:
                    self.config["budget"] = b
                    console.print(f"[green]Budget ceiling set to ${b:.4f}[/green]")
            except ValueError:
                console.print("[red]Invalid number for budget.[/red]")

        elif param == "quality":
            if val in ("low", "medium", "high"):
                self.config["quality_requirement"] = val
                console.print(f"[green]Quality floor set to '{val}'[/green]")
            else:
                console.print("[red]Invalid quality floor. Choose: low, medium, high[/red]")

        elif param == "slo":
            try:
                slo = float(val)
                if slo <= 0:
                    console.print("[red]SLO must be greater than 0.[/red]")
                else:
                    self.config["latency_slo"] = slo
                    console.print(f"[green]Latency SLO set to {slo}s[/green]")
            except ValueError:
                console.print("[red]Invalid number for latency SLO.[/red]")

        elif param == "tokens":
            try:
                t = int(val)
                self.config["max_tokens"] = t
                console.print(f"[green]Max tokens set to {t}[/green]")
            except ValueError:
                console.print("[red]Invalid integer for max tokens.[/red]")

        elif param == "policy":
            if val in ("adaptive", "strongest", "cheapest", "static"):
                self.config["policy"] = val
                console.print(f"[green]Policy set to '{val}'[/green]")
            else:
                console.print("[red]Invalid policy. Choose: adaptive, strongest, cheapest, static[/red]")

        elif param == "scheduler":
            if val in ("slo_aware", "least_loaded", "round_robin"):
                self.config["scheduler"] = val
                console.print(f"[green]Scheduler set to '{val}'[/green]")
            else:
                console.print("[red]Invalid scheduler. Choose: slo_aware, least_loaded, round_robin[/red]")

        elif param in ("history", "hist"):
            if val in ("on", "true", "yes", "1"):
                self.config["history"] = True
                console.print("[green]Conversation history ENABLED (multi-turn chat)[/green]")
            else:
                self.config["history"] = False
                self.messages.clear()
                console.print("[dim]Conversation history DISABLED (single-query workbench mode)[/dim]")

        elif param in ("private", "user_private"):
            if val in ("on", "true", "yes", "1"):
                self.config["user_private"] = True
                console.print("[yellow]User Private toggle ENABLED (local self-hosted models only)[/yellow]")
            else:
                self.config["user_private"] = False
                console.print("[green]User Private toggle DISABLED (cloud models eligible)[/green]")

        elif param == "cache":
            if val in ("off", "false", "bypass"):
                self.config["no_cache"] = True
                console.print("[yellow]Cache bypassed (no_cache=True)[/yellow]")
            else:
                self.config["no_cache"] = False
                console.print("[green]Cache enabled (no_cache=False)[/green]")

        elif param == "url":
            self.client.set_base_url(args[1])
            console.print(f"[green]Gateway URL set to '{self.client.base_url}'[/green]")

        else:
            console.print(f"[red]Unknown parameter '{param}'. Type /help for options.[/red]")

    def run(self) -> None:
        """Main interactive loop."""
        self.print_header()

        while True:
            try:
                console.print()
                user_input = console.input("[bold blue]user > [/bold blue]").strip()
                if not user_input:
                    continue

                # Handle slash commands
                if user_input.startswith("/"):
                    parts = user_input.split()
                    cmd = parts[0].lower()
                    args = parts[1:]

                    if cmd in ("/exit", "/quit", "/q"):
                        console.print("[dim]Goodbye![/dim]")
                        sys.exit(0)
                    elif cmd in ("/help", "/h", "/?"):
                        self.print_help()
                    elif cmd in ("/settings", "/config"):
                        self.print_settings()
                    elif cmd == "/set":
                        self.handle_set(args)
                    elif cmd in ("/history", "/hist"):
                        if args:
                            self.config["history"] = args[0].lower() in ("true", "1", "on", "yes")
                        else:
                            self.config["history"] = not self.config["history"]
                        state = (
                            "[bold green]ENABLED (multi-turn chat)[/bold green]"
                            if self.config["history"]
                            else "[dim]DISABLED (single-query workbench mode)[/dim]"
                        )
                        console.print(f"Conversation history mode is now {state}")
                        if not self.config["history"]:
                            self.messages.clear()
                    elif cmd in ("/private", "/user_private"):
                        self.config["user_private"] = not self.config["user_private"]
                        state = (
                            "[bold yellow]ENABLED (local only)[/bold yellow]"
                            if self.config["user_private"]
                            else "[bold green]DISABLED (cloud allowed)[/bold green]"
                        )
                        console.print(f"User Private mode is now {state}")
                    elif cmd in ("/diag", "/diagnostics"):
                        self.print_diagnostics()
                    elif cmd == "/health":
                        self.check_and_print_health()
                    elif cmd in ("/clear", "/cls"):
                        console.clear()
                        self.messages.clear()
                        self.print_header()
                    else:
                        console.print(f"[red]Unknown command '{cmd}'. Type /help for available commands.[/red]")
                    continue

                # Normal prompt dispatch
                if self.config["history"]:
                    self.messages.append({"role": "user", "content": user_input})
                    send_messages = list(self.messages)
                else:
                    send_messages = [{"role": "user", "content": user_input}]

                with console.status("[bold cyan]Routing through AdaRoute Gateway...[/bold cyan]", spinner="dots"):
                    res = self.client.send_chat(
                        messages=send_messages,
                        capability=self.config["capability"],
                        privacy_level=self.config["privacy_level"],
                        budget=self.config["budget"],
                        quality_requirement=self.config["quality_requirement"],
                        latency_slo=self.config["latency_slo"],
                        max_tokens=self.config["max_tokens"],
                        policy=self.config["policy"],
                        scheduler=self.config["scheduler"],
                        tenant_id=self.config["tenant_id"],
                        user_id=self.config["user_id"],
                        no_cache=self.config["no_cache"],
                        user_private=self.config["user_private"],
                    )

                if res.get("success"):
                    content = res.get("content", "")
                    meta = res.get("routing_metadata", {})
                    self.last_metadata = meta
                    if self.config["history"]:
                        self.messages.append({"role": "assistant", "content": content})

                    # Render Assistant response panel
                    console.print(
                        Panel(
                            content,
                            title="[bold green]AdaRoute Assistant[/bold green]",
                            border_style="green",
                            box=box.ROUNDED,
                        )
                    )

                    # Render mini-telemetry bar
                    cache_status = (
                        "[bold green]CACHE HIT[/bold green]" if meta.get("cache_hit") else "[dim]CACHE MISS[/dim]"
                    )
                    slo_icon = "✅" if meta.get("slo_attained", True) else "⚠️"
                    console.print(
                        f"[dim]Model:[/dim] [cyan]{meta.get('selected_model')}[/cyan] "
                        f"[dim]Replica:[/dim] [yellow]{meta.get('selected_replica')}[/yellow] "
                        f"[dim]Latency:[/dim] {meta.get('latency_ms', 0):.1f}ms "
                        f"[dim]Cost:[/dim] ${meta.get('actual_cost', 0):.6f} "
                        f"{cache_status} "
                        f"[dim]SLO:[/dim] {slo_icon}  "
                        f"[dim](type [bold white]/diag[/bold white] for details)[/dim]"
                    )
                else:
                    status_code = res.get("status_code", 0)
                    error_msg = res.get("error", "Unknown error")
                    console.print(
                        Panel(
                            f"[bold red]HTTP {status_code} Error:[/bold red]\n{error_msg}",
                            title="[bold red]Gateway Error[/bold red]",
                            border_style="red",
                            box=box.ROUNDED,
                        )
                    )

            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Session terminated.[/dim]")
                break


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h"):
        console.print("[bold cyan]AdaRoute Terminal Workbench CLI[/bold cyan]")
        console.print("Usage: python3 ui/cli/main.py [GATEWAY_URL]")
        console.print("Default GATEWAY_URL: http://localhost:8000\n")
        console.print("Interactive Commands:")
        console.print("  /help                     Show available commands")
        console.print("  /history [on|off]         Toggle conversation history (default: OFF for caching tests)")
        console.print("  /private                  Toggle user_private mode (local models only)")
        console.print("  /set <key> <val>          Update configuration (capability, privacy, budget, etc.)")
        console.print("  /diag                     Show last routing telemetry and decision tree")
        console.print("  /clear                    Reset chat context")
        console.print("  /exit                     Exit the workbench")
        return

    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    cli = AdaRouteCLI(base_url=base_url)
    cli.run()


if __name__ == "__main__":
    main()
