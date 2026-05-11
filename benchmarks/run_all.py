"""Run all benchmark suites and produce unified report.

Usage:
    python benchmarks/run_all.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def main():
    results_dir = Path("benchmarks/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  qufin Comprehensive Benchmark Suite")
    print("=" * 70)
    total_t0 = time.perf_counter()

    # 1. Classical comparison
    print("\n[1/3] Classical Baseline Comparisons")
    print("-" * 40)
    from benchmarks.classical_comparison import run_all as run_classical, to_markdown as md_classical
    from dataclasses import asdict
    classical_entries = run_classical()
    with open(results_dir / "classical.json", "w") as f:
        json.dump([asdict(e) for e in classical_entries], f, indent=2)
    (results_dir / "classical.md").write_text(md_classical(classical_entries))
    print(f"  -> {len(classical_entries)} entries")

    # 2. Quantum scaling
    print("\n[2/3] Quantum Scaling Analysis")
    print("-" * 40)
    from benchmarks.quantum_scaling import run_all as run_quantum, to_markdown as md_quantum
    quantum_entries = run_quantum()
    with open(results_dir / "quantum_scaling.json", "w") as f:
        json.dump([asdict(e) for e in quantum_entries], f, indent=2)
    (results_dir / "quantum_scaling.md").write_text(md_quantum(quantum_entries))
    print(f"  -> {len(quantum_entries)} entries")

    # 3. Quantum advantage
    print("\n[3/3] Quantum Advantage Analysis")
    print("-" * 40)
    from benchmarks.quantum_advantage import run_all as run_advantage, to_markdown as md_advantage
    advantage_analyses = run_advantage()
    data = [
        {"title": a.title, "summary": a.summary, "estimates": [asdict(e) for e in a.estimates]}
        for a in advantage_analyses
    ]
    with open(results_dir / "advantage.json", "w") as f:
        json.dump(data, f, indent=2)
    (results_dir / "advantage.md").write_text(md_advantage(advantage_analyses))
    print(f"  -> {len(advantage_analyses)} analyses")

    total_time = time.perf_counter() - total_t0

    # Summary
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Classical benchmarks: {len(classical_entries)} entries")
    print(f"  Quantum scaling:      {len(quantum_entries)} entries")
    print(f"  Advantage analyses:   {len(advantage_analyses)} reports")
    print(f"  Total wall time:      {total_time:.1f}s")
    print(f"\n  Results saved to: {results_dir.absolute()}/")
    print(f"    - classical.json + classical.md")
    print(f"    - quantum_scaling.json + quantum_scaling.md")
    print(f"    - advantage.json + advantage.md")


if __name__ == "__main__":
    main()
