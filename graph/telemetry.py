from __future__ import annotations


def record_node(
    state: dict,
    name: str,
    duration_ms: float,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
) -> list[dict]:
    entry = {
        "node": name,
        "duration_ms": duration_ms,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": round(cost_usd, 6),
    }
    return [*state.get("node_trace", []), entry]
