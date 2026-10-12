"""AdaRoute CLI HTTP Client.
Communicates strictly over HTTP with AdaRoute Gateway.
"""

from typing import Any

import httpx


class GatewayCLIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def set_base_url(self, url: str) -> None:
        self.base_url = url.rstrip("/")

    def check_health(self) -> dict[str, Any]:
        """Queries GET /health."""
        try:
            with httpx.Client(timeout=4.0) as client:
                res = client.get(f"{self.base_url}/health")
                if res.status_code == 200:
                    return res.json()
                return {
                    "status": "unavailable",
                    "error": f"HTTP {res.status_code}: {res.text}",
                }
        except Exception as e:
            return {"status": "unavailable", "error": str(e)}

    def send_chat(
        self,
        messages: list[dict[str, str]],
        capability: str = "general",
        privacy_level: str = "public",
        budget: float = 1.0,
        quality_requirement: str = "medium",
        latency_slo: float = 2.0,
        max_tokens: int = 256,
        policy: str = "adaptive",
        scheduler: str = "slo_aware",
        tenant_id: str = "user",
        user_id: str = "user",
        no_cache: bool = False,
        user_private: bool = False,
    ) -> dict[str, Any]:
        """Dispatches chat completion request to POST /v1/chat/completions."""
        payload = {
            "model": capability,
            "messages": messages,
            "privacy_level": privacy_level,
            "budget": budget,
            "quality_requirement": quality_requirement,
            "latency_slo": latency_slo,
            "max_tokens": max_tokens,
            "policy": policy,
            "scheduler": scheduler,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "no_cache": no_cache,
            "user_private": user_private,
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                res = client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )

                if res.status_code == 200:
                    data = res.json()
                    content = data.get("content")
                    if not content and data.get("choices"):
                        content = data["choices"][0].get("message", {}).get("content", "")
                    return {
                        "success": True,
                        "content": content,
                        "routing_metadata": data.get("routing_metadata", {}),
                        "usage": data.get("usage", {}),
                    }

                # Parse detailed error
                try:
                    err_json = res.json()
                    detail = err_json.get("detail", res.text)
                except Exception:
                    detail = res.text

                return {
                    "success": False,
                    "status_code": res.status_code,
                    "error": detail,
                }
        except httpx.ConnectError:
            return {
                "success": False,
                "status_code": 0,
                "error": f"Failed to connect to AdaRoute Gateway at {self.base_url}. Is the service running?",
            }
        except httpx.TimeoutException:
            return {
                "success": False,
                "status_code": 408,
                "error": "Request timed out waiting for gateway response.",
            }
        except Exception as e:
            return {
                "success": False,
                "status_code": 500,
                "error": f"Unexpected client error: {e!s}",
            }
