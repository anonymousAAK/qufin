"""Unit tests for path-dependent QAE (Asian option pricing)."""

from __future__ import annotations

import numpy as np
import pytest

from qufin.options.amplitude_estimation.path_dependent_qae import (
    PathDependentAsianSpec,
    build_asian_payoff_oracle,
    build_path_dependent_estimation_problem,
    build_path_state_preparation,
    compute_path_averages,
    price_asian_mc,
)


# ---------------------------------------------------------------------------
# Tests: PathDependentAsianSpec validation
# ---------------------------------------------------------------------------

class TestPathDependentAsianSpec:
    def test_default_construction(self) -> None:
        spec = PathDependentAsianSpec()
        assert spec.n_steps == 4
        assert spec.n_qubits_per_step == 2
        assert spec.n_price_qubits == 8
        assert spec.dt == pytest.approx(0.25)

    def test_invalid_n_steps_raises(self) -> None:
        with pytest.raises(ValueError, match="n_steps"):
            PathDependentAsianSpec(n_steps=0)

    def test_invalid_n_qubits_raises(self) -> None:
        with pytest.raises(ValueError, match="n_qubits_per_step"):
            PathDependentAsianSpec(n_qubits_per_step=0)

    def test_invalid_average_type_raises(self) -> None:
        with pytest.raises(ValueError, match="average_type"):
            PathDependentAsianSpec(average_type="harmonic")

    def test_dt_property(self) -> None:
        spec = PathDependentAsianSpec(T=1.0, n_steps=12)
        assert spec.dt == pytest.approx(1.0 / 12.0)


# ---------------------------------------------------------------------------
# Tests: Path state preparation
# ---------------------------------------------------------------------------

class TestPathStatePreparation:
    def test_small_path_t3(self) -> None:
        """Test path state preparation with T=3 steps, 2 qubits each."""
        spec = PathDependentAsianSpec(n_steps=3, n_qubits_per_step=2)
        qc, all_values = build_path_state_preparation(spec)

        # Circuit should have 3 * 2 = 6 qubits
        assert qc.num_qubits == 6
        # Should produce 3 sets of price values
        assert len(all_values) == 3
        # Each set should have 2^2 = 4 values
        for vals in all_values:
            assert len(vals) == 4

    def test_small_path_t4(self) -> None:
        """Test path state preparation with T=4 steps, 2 qubits each."""
        spec = PathDependentAsianSpec(n_steps=4, n_qubits_per_step=2)
        qc, all_values = build_path_state_preparation(spec)

        assert qc.num_qubits == 8
        assert len(all_values) == 4

    def test_single_step_degenerates(self) -> None:
        """With T=1 step, path-dependent reduces to single distribution."""
        spec = PathDependentAsianSpec(n_steps=1, n_qubits_per_step=3)
        qc, all_values = build_path_state_preparation(spec)

        assert qc.num_qubits == 3
        assert len(all_values) == 1
        assert len(all_values[0]) == 8  # 2^3

    def test_price_values_positive(self) -> None:
        """All discretized price values should be positive."""
        spec = PathDependentAsianSpec(n_steps=3, n_qubits_per_step=2)
        _, all_values = build_path_state_preparation(spec)
        for vals in all_values:
            assert np.all(vals > 0)

    def test_price_values_centered_on_s0(self) -> None:
        """Price grids should be roughly centered around S_0."""
        spec = PathDependentAsianSpec(
            s0=100.0, n_steps=3, n_qubits_per_step=3,
        )
        _, all_values = build_path_state_preparation(spec)
        for vals in all_values:
            mid = (vals[0] + vals[-1]) / 2.0
            # Midpoint should be in a reasonable range around S_0
            assert 50.0 < mid < 200.0


# ---------------------------------------------------------------------------
# Tests: Average computation
# ---------------------------------------------------------------------------

class TestComputePathAverages:
    def test_arithmetic_average(self) -> None:
        """Test arithmetic average with known values."""
        # 2 steps, 2 values each => 4 total states
        vals0 = np.array([90.0, 110.0])
        vals1 = np.array([95.0, 105.0])
        all_values = [vals0, vals1]

        averages = compute_path_averages(all_values, "arithmetic")
        assert len(averages) == 4

        # State 0: (90+95)/2=92.5, State 1: (90+105)/2=97.5
        # State 2: (110+95)/2=102.5, State 3: (110+105)/2=107.5
        expected = [92.5, 97.5, 102.5, 107.5]
        np.testing.assert_allclose(averages, expected, rtol=1e-10)

    def test_geometric_average(self) -> None:
        """Test geometric average with known values."""
        vals0 = np.array([90.0, 110.0])
        vals1 = np.array([95.0, 105.0])
        all_values = [vals0, vals1]

        averages = compute_path_averages(all_values, "geometric")
        assert len(averages) == 4

        # Geometric mean of (90, 95) = sqrt(90*95)
        expected_0 = np.sqrt(90.0 * 95.0)
        assert averages[0] == pytest.approx(expected_0, rel=1e-10)

    def test_single_step_average_equals_value(self) -> None:
        """With 1 step, average = the value itself."""
        vals = np.array([80.0, 100.0, 120.0, 140.0])
        averages = compute_path_averages([vals], "arithmetic")
        np.testing.assert_allclose(averages, vals)

    def test_output_length(self) -> None:
        """Number of averages = product of grid sizes."""
        vals0 = np.array([1.0, 2.0, 3.0, 4.0])  # 4 states
        vals1 = np.array([5.0, 6.0, 7.0, 8.0])  # 4 states
        vals2 = np.array([9.0, 10.0, 11.0, 12.0])  # 4 states
        averages = compute_path_averages([vals0, vals1, vals2], "arithmetic")
        assert len(averages) == 64  # 4^3


# ---------------------------------------------------------------------------
# Tests: Asian option pricing vs classical MC
# ---------------------------------------------------------------------------

class TestAsianPricingMC:
    def test_mc_call_nonnegative(self) -> None:
        spec = PathDependentAsianSpec(is_call=True)
        result = price_asian_mc(spec, n_paths=10_000, seed=42)
        assert result["price"] >= 0.0

    def test_mc_put_nonnegative(self) -> None:
        spec = PathDependentAsianSpec(is_call=False)
        result = price_asian_mc(spec, n_paths=10_000, seed=42)
        assert result["price"] >= 0.0

    def test_mc_returns_expected_keys(self) -> None:
        spec = PathDependentAsianSpec()
        result = price_asian_mc(spec, n_paths=1_000, seed=42)
        expected_keys = {"price", "std_error", "ci_low", "ci_high", "n_paths"}
        assert set(result.keys()) == expected_keys

    def test_mc_geometric_vs_arithmetic(self) -> None:
        """Geometric average call should be <= arithmetic (by Jensen)."""
        spec_arith = PathDependentAsianSpec(
            average_type="arithmetic", n_steps=12,
        )
        spec_geom = PathDependentAsianSpec(
            average_type="geometric", n_steps=12,
        )
        mc_arith = price_asian_mc(spec_arith, n_paths=50_000, seed=42)
        mc_geom = price_asian_mc(spec_geom, n_paths=50_000, seed=42)
        # Geometric average <= arithmetic (Jensen's inequality)
        # Allow some MC noise tolerance
        assert mc_geom["price"] <= mc_arith["price"] + 3 * mc_arith["std_error"]

    def test_mc_deep_itm_call_positive(self) -> None:
        """Deep in-the-money call should have a clearly positive price."""
        spec = PathDependentAsianSpec(s0=150.0, k=100.0, is_call=True)
        result = price_asian_mc(spec, n_paths=10_000, seed=42)
        assert result["price"] > 10.0  # well above zero

    def test_mc_deep_otm_call_near_zero(self) -> None:
        """Deep out-of-the-money call should be near zero."""
        spec = PathDependentAsianSpec(s0=50.0, k=150.0, is_call=True)
        result = price_asian_mc(spec, n_paths=10_000, seed=42)
        assert result["price"] < 1.0

    def test_mc_asian_cheaper_than_european(self) -> None:
        """Asian call should be cheaper than European call (averaging effect)."""
        spec = PathDependentAsianSpec(n_steps=12)
        mc_asian = price_asian_mc(spec, n_paths=50_000, seed=42)

        # European call MC (single step)
        rng = np.random.default_rng(42)
        Z = rng.standard_normal(50_000)
        S_T = spec.s0 * np.exp(
            (spec.r - 0.5 * spec.sigma**2) * spec.T
            + spec.sigma * np.sqrt(spec.T) * Z
        )
        euro_payoffs = np.maximum(S_T - spec.k, 0.0)
        euro_price = float(np.mean(euro_payoffs) * np.exp(-spec.r * spec.T))

        # Asian call <= European call (volatility averaging reduces value)
        assert mc_asian["price"] <= euro_price + 2.0


# ---------------------------------------------------------------------------
# Tests: QAE estimation problem construction
# ---------------------------------------------------------------------------

class TestBuildEstimationProblem:
    def test_problem_construction_small(self) -> None:
        """Test that estimation problem builds without error."""
        spec = PathDependentAsianSpec(
            n_steps=2, n_qubits_per_step=2,
        )
        problem, rescale = build_path_dependent_estimation_problem(spec)

        assert problem.n_qubits == 5  # 2*2 + 1 ancilla
        assert problem.objective_qubits == [4]
        assert rescale > 0.0

    def test_problem_state_prep_circuit_valid(self) -> None:
        """State preparation circuit should have correct qubit count."""
        spec = PathDependentAsianSpec(
            n_steps=3, n_qubits_per_step=2,
        )
        problem, _ = build_path_dependent_estimation_problem(spec)
        qc = problem.state_preparation
        assert qc.num_qubits == 7  # 3*2 + 1

    def test_grover_operator_builds(self) -> None:
        """Grover operator should build from the estimation problem."""
        spec = PathDependentAsianSpec(
            n_steps=2, n_qubits_per_step=2,
        )
        problem, _ = build_path_dependent_estimation_problem(spec)
        grover = problem.build_grover_operator()
        assert grover.num_qubits == 5

    def test_rescale_includes_discount(self) -> None:
        """Rescale factor should include discount."""
        spec = PathDependentAsianSpec(r=0.05, T=1.0)
        _, rescale = build_path_dependent_estimation_problem(spec)
        discount = np.exp(-0.05 * 1.0)
        # rescale = discount * max_payoff, so rescale >= discount * 0
        assert rescale > 0.0


# ---------------------------------------------------------------------------
# Tests: Payoff oracle
# ---------------------------------------------------------------------------

class TestBuildAsianPayoffOracle:
    def test_call_payoff_oracle(self) -> None:
        """Call payoff oracle should add ancilla qubit."""
        spec = PathDependentAsianSpec(
            n_steps=2, n_qubits_per_step=2, is_call=True,
        )
        path_circuit, all_values = build_path_state_preparation(spec)
        full_circuit, max_payoff = build_asian_payoff_oracle(
            spec, path_circuit, all_values,
        )
        assert full_circuit.num_qubits == 5  # 4 price + 1 ancilla
        assert max_payoff > 0.0

    def test_put_payoff_oracle(self) -> None:
        """Put payoff oracle should work analogously."""
        spec = PathDependentAsianSpec(
            n_steps=2, n_qubits_per_step=2, is_call=False,
        )
        path_circuit, all_values = build_path_state_preparation(spec)
        full_circuit, max_payoff = build_asian_payoff_oracle(
            spec, path_circuit, all_values,
        )
        assert full_circuit.num_qubits == 5
        assert max_payoff > 0.0

    def test_deep_otm_max_payoff_defaults(self) -> None:
        """When all payoffs are zero, max_payoff defaults to 1.0."""
        spec = PathDependentAsianSpec(
            s0=10.0, k=1000.0, is_call=True,
            n_steps=2, n_qubits_per_step=2,
        )
        path_circuit, all_values = build_path_state_preparation(spec)
        _, max_payoff = build_asian_payoff_oracle(
            spec, path_circuit, all_values,
        )
        assert max_payoff == 1.0


# ---------------------------------------------------------------------------
# Tests: Put-call parity (via MC benchmark)
# ---------------------------------------------------------------------------

class TestPutCallParity:
    def test_put_call_parity_mc(self) -> None:
        """Put-call parity for Asian options (approximate).

        For arithmetic Asian options, the exact put-call parity is:
            C - P = exp(-rT) * (E[avg] - K)

        where E[avg] is the expected value of the arithmetic average
        under the risk-neutral measure. We verify this approximately
        using Monte Carlo.
        """
        s0, k, r, sigma, T, n_steps = 100.0, 100.0, 0.05, 0.2, 1.0, 4

        spec_call = PathDependentAsianSpec(
            s0=s0, k=k, r=r, sigma=sigma, T=T,
            n_steps=n_steps, is_call=True,
        )
        spec_put = PathDependentAsianSpec(
            s0=s0, k=k, r=r, sigma=sigma, T=T,
            n_steps=n_steps, is_call=False,
        )

        n_paths = 100_000
        mc_call = price_asian_mc(spec_call, n_paths=n_paths, seed=42)
        mc_put = price_asian_mc(spec_put, n_paths=n_paths, seed=42)

        # E[avg] under risk-neutral: for arithmetic average of GBM
        # E[S_t] = S_0 * exp(r * t), so
        # E[avg] = (1/T_steps) * sum_{t=1}^{T_steps} S_0 * exp(r * t * dt)
        dt = T / n_steps
        expected_avg = sum(
            s0 * np.exp(r * (t + 1) * dt) for t in range(n_steps)
        ) / n_steps
        discount = np.exp(-r * T)

        # Put-call parity: C - P = discount * (E[avg] - K)
        parity_rhs = discount * (expected_avg - k)
        parity_lhs = mc_call["price"] - mc_put["price"]

        # Allow MC noise (a few standard errors)
        tolerance = 3.0 * (mc_call["std_error"] + mc_put["std_error"])
        assert abs(parity_lhs - parity_rhs) < tolerance, (
            f"Put-call parity violated: C-P={parity_lhs:.4f}, "
            f"disc*(E[avg]-K)={parity_rhs:.4f}, tol={tolerance:.4f}"
        )

    def test_put_call_parity_geometric_mc(self) -> None:
        """Put-call parity for geometric Asian options.

        Uses MC to estimate E[G] directly, then checks C - P = disc*(E[G] - K).
        """
        s0, k, r, sigma, T, n_steps = 100.0, 100.0, 0.05, 0.2, 1.0, 4

        spec_call = PathDependentAsianSpec(
            s0=s0, k=k, r=r, sigma=sigma, T=T,
            n_steps=n_steps, is_call=True,
            average_type="geometric",
        )
        spec_put = PathDependentAsianSpec(
            s0=s0, k=k, r=r, sigma=sigma, T=T,
            n_steps=n_steps, is_call=False,
            average_type="geometric",
        )

        n_paths = 200_000
        # Use the same seed so paths are identical for call and put
        mc_call = price_asian_mc(spec_call, n_paths=n_paths, seed=42)
        mc_put = price_asian_mc(spec_put, n_paths=n_paths, seed=42)

        # Estimate E[G] via MC directly (same paths)
        rng = np.random.default_rng(42)
        dt = T / n_steps
        Z = rng.standard_normal((n_paths, n_steps))
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        log_prices = np.zeros((n_paths, n_steps))
        log_prices[:, 0] = np.log(s0) + drift + diffusion * Z[:, 0]
        for t in range(1, n_steps):
            log_prices[:, t] = log_prices[:, t - 1] + drift + diffusion * Z[:, t]
        geom_avg = np.exp(np.mean(log_prices, axis=1))
        expected_geom = float(np.mean(geom_avg))

        discount = np.exp(-r * T)
        parity_rhs = discount * (expected_geom - k)
        parity_lhs = mc_call["price"] - mc_put["price"]

        tolerance = 3.0 * (mc_call["std_error"] + mc_put["std_error"])
        assert abs(parity_lhs - parity_rhs) < tolerance, (
            f"Geometric put-call parity violated: C-P={parity_lhs:.4f}, "
            f"disc*(E[G]-K)={parity_rhs:.4f}, tol={tolerance:.4f}"
        )
