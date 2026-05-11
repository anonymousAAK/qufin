"""Benchmark leaderboard: tabular summary of solver results.

Generates Markdown or CSV leaderboards from BenchmarkRow results,
ranked by objective value or approximation ratio.
"""

from __future__ import annotations

import csv
import io
from typing import Literal

from qufin.benchmarks.runner import BenchmarkRow


def _format_float(v: float | None, precision: int = 6) -> str:
    if v is None:
        return "—"
    return f"{v:.{precision}f}"


def generate_leaderboard(
    rows: list[BenchmarkRow],
    sort_by: Literal["objective", "wall_seconds", "rel_error"] = "objective",
    ascending: bool = True,
) -> list[dict[str, str]]:
    """Generate a leaderboard table from benchmark rows.

    Returns list of row dicts with string-formatted values.
    """
    def key_fn(r):
        return getattr(r, sort_by) if getattr(r, sort_by) is not None else float("inf")
    sorted_rows = sorted(rows, key=key_fn, reverse=not ascending)

    table = []
    for rank, row in enumerate(sorted_rows, 1):
        table.append({
            "rank": str(rank),
            "problem": row.problem_id,
            "solver": row.solver_name,
            "family": row.solver_family,
            "objective": _format_float(row.objective),
            "rel_error": _format_float(row.rel_error, 4),
            "wall_s": _format_float(row.wall_seconds, 3),
            "qubits": str(row.n_qubits) if row.n_qubits else "—",
            "depth": str(row.circuit_depth) if row.circuit_depth else "—",
            "backend": row.backend or "—",
        })
    return table


def to_markdown(rows: list[BenchmarkRow], **kwargs) -> str:
    """Generate a Markdown leaderboard table."""
    table = generate_leaderboard(rows, **kwargs)
    if not table:
        return "No results."

    headers = list(table[0].keys())
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in table:
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")
    return "\n".join(lines)


def to_csv(rows: list[BenchmarkRow], **kwargs) -> str:
    """Generate a CSV leaderboard."""
    table = generate_leaderboard(rows, **kwargs)
    if not table:
        return ""

    output = io.StringIO()
    headers = list(table[0].keys())
    writer = csv.DictWriter(output, fieldnames=headers)
    writer.writeheader()
    writer.writerows(table)
    return output.getvalue()
