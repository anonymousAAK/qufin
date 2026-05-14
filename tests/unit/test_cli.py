"""Tests for the qufin CLI module."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from qufin.cli import (
    _SUPPORTED_FORMATS,
    CLIResult,
    build_parser,
    handle_benchmark,
    handle_optimize,
    handle_price,
    handle_risk,
    install_completion,
    main,
    write_output,
)

# ------------------------------------------------------------------
# CLIResult
# ------------------------------------------------------------------


class TestCLIResult:
    """Tests for CLIResult dataclass."""

    def test_default_construction(self):
        r = CLIResult()
        assert r.command == ""
        assert r.data == {}
        assert r.errors == []

    def test_to_dict(self):
        r = CLIResult(command="test", data={"a": 1}, errors=["warn"])
        d = r.to_dict()
        assert d["command"] == "test"
        assert d["data"] == {"a": 1}
        assert d["errors"] == ["warn"]

    def test_to_dict_roundtrip_json(self):
        r = CLIResult(command="price", data={"spot": 100.0})
        text = json.dumps(r.to_dict())
        loaded = json.loads(text)
        assert loaded["command"] == "price"
        assert loaded["data"]["spot"] == 100.0


# ------------------------------------------------------------------
# Parser construction
# ------------------------------------------------------------------


class TestBuildParser:
    """Tests for the argparse parser builder."""

    def test_parser_has_version_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--version"])
        assert args.version is True

    def test_parser_format_default(self):
        parser = build_parser()
        args = parser.parse_args(["optimize", "--universe", "sp500", "--method", "qaoa"])
        assert args.format == "json"

    def test_parser_format_csv(self):
        parser = build_parser()
        args = parser.parse_args([
            "--format", "csv", "optimize", "--universe", "sp500", "--method", "qaoa",
        ])
        assert args.format == "csv"

    def test_optimize_subcommand_parsing(self):
        parser = build_parser()
        args = parser.parse_args([
            "optimize", "--universe", "sp500", "--method", "qaoa",
            "--cardinality", "20", "--budget", "0.8",
        ])
        assert args.command == "optimize"
        assert args.universe == "sp500"
        assert args.method == "qaoa"
        assert args.cardinality == 20
        assert args.budget == 0.8

    def test_price_subcommand_parsing(self):
        parser = build_parser()
        args = parser.parse_args([
            "price", "--type", "european", "--s", "100", "--k", "105",
            "--method", "iqae",
        ])
        assert args.command == "price"
        assert args.type == "european"
        assert args.s == 100.0
        assert args.k == 105.0
        assert args.method == "iqae"

    def test_risk_subcommand_parsing(self):
        parser = build_parser()
        args = parser.parse_args([
            "risk", "--portfolio", "port.csv", "--method", "quantum_var",
            "--alpha", "0.99",
        ])
        assert args.command == "risk"
        assert args.portfolio == "port.csv"
        assert args.method == "quantum_var"
        assert args.alpha == 0.99

    def test_benchmark_subcommand_parsing(self):
        parser = build_parser()
        args = parser.parse_args([
            "benchmark", "--problem", "15",
            "--algorithms", "qaoa,vqe,mvo",
            "--output", "results.json",
        ])
        assert args.command == "benchmark"
        assert args.problem == 15
        assert args.algorithms == "qaoa,vqe,mvo"

    def test_price_defaults(self):
        parser = build_parser()
        args = parser.parse_args([
            "price", "--type", "european", "--s", "100", "--k", "105",
            "--method", "black_scholes",
        ])
        assert args.r == 0.05
        assert args.sigma == 0.2
        assert args.T == 1.0

    def test_optimize_cardinality_default(self):
        parser = build_parser()
        args = parser.parse_args([
            "optimize", "--universe", "sp500", "--method", "qaoa",
        ])
        assert args.cardinality == 10


# ------------------------------------------------------------------
# Subcommand handlers
# ------------------------------------------------------------------


class TestHandleOptimize:
    """Tests for the optimize handler."""

    def test_qaoa_method(self):
        parser = build_parser()
        args = parser.parse_args([
            "optimize", "--universe", "sp500", "--method", "qaoa",
            "--cardinality", "20",
        ])
        result = handle_optimize(args)
        assert result.command == "optimize"
        assert result.data["method"] == "qaoa"
        assert result.data["solver"] == "quantum"
        assert result.data["status"] == "completed"
        assert "20" in result.data["description"]

    def test_mvo_method(self):
        parser = build_parser()
        args = parser.parse_args([
            "optimize", "--universe", "custom", "--method", "mvo",
        ])
        result = handle_optimize(args)
        assert result.data["solver"] == "classical"

    def test_vqe_method(self):
        parser = build_parser()
        args = parser.parse_args([
            "optimize", "--universe", "sp500", "--method", "vqe",
        ])
        result = handle_optimize(args)
        assert result.data["solver"] == "quantum"


class TestHandlePrice:
    """Tests for the price handler."""

    def test_iqae_quantum(self):
        parser = build_parser()
        args = parser.parse_args([
            "price", "--type", "european", "--s", "100", "--k", "105",
            "--method", "iqae",
        ])
        result = handle_price(args)
        assert result.command == "price"
        assert result.data["solver"] == "quantum"
        assert result.data["spot"] == 100.0
        assert result.data["strike"] == 105.0

    def test_black_scholes(self):
        parser = build_parser()
        args = parser.parse_args([
            "price", "--type", "european", "--s", "100", "--k", "100",
            "--method", "black_scholes",
        ])
        result = handle_price(args)
        assert result.data["solver"] == "analytical"

    def test_monte_carlo(self):
        parser = build_parser()
        args = parser.parse_args([
            "price", "--type", "asian", "--s", "50", "--k", "55",
            "--method", "monte_carlo",
        ])
        result = handle_price(args)
        assert result.data["solver"] == "classical"


class TestHandleRisk:
    """Tests for the risk handler."""

    def test_quantum_var(self):
        parser = build_parser()
        args = parser.parse_args([
            "risk", "--portfolio", "port.csv", "--method", "quantum_var",
            "--alpha", "0.99",
        ])
        result = handle_risk(args)
        assert result.command == "risk"
        assert result.data["solver"] == "quantum"
        assert result.data["alpha"] == 0.99

    def test_historical_var(self):
        parser = build_parser()
        args = parser.parse_args([
            "risk", "--portfolio", "port.csv", "--method", "historical",
        ])
        result = handle_risk(args)
        assert result.data["solver"] == "classical"


class TestHandleBenchmark:
    """Tests for the benchmark handler."""

    def test_basic_benchmark(self):
        parser = build_parser()
        args = parser.parse_args([
            "benchmark", "--problem", "15",
            "--algorithms", "qaoa,vqe,mvo",
        ])
        result = handle_benchmark(args)
        assert result.command == "benchmark"
        assert result.data["algorithms"] == ["qaoa", "vqe", "mvo"]
        assert result.data["problem_size"] == 15


# ------------------------------------------------------------------
# Output formatting
# ------------------------------------------------------------------


class TestWriteOutput:
    """Tests for output serialization."""

    def test_json_output_string(self):
        r = CLIResult(command="test", data={"val": 42})
        text = write_output(r, output_path=None, fmt="json")
        loaded = json.loads(text)
        assert loaded["data"]["val"] == 42

    def test_json_output_file(self):
        r = CLIResult(command="test", data={"val": 1})
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            write_output(r, output_path=path, fmt="json")
            with open(path) as fh:
                loaded = json.load(fh)
            assert loaded["data"]["val"] == 1
        finally:
            os.unlink(path)

    def test_csv_output_string(self):
        r = CLIResult(command="test", data={"a": 1, "b": 2})
        text = write_output(r, output_path=None, fmt="csv")
        assert "a" in text
        assert "b" in text

    def test_unsupported_format_raises(self):
        r = CLIResult(command="test")
        with pytest.raises(ValueError, match="Unsupported format"):
            write_output(r, output_path=None, fmt="xml")

    def test_parquet_output(self):
        r = CLIResult(command="test", data={"x": 10, "y": 20})
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name
        try:
            result_path = write_output(r, output_path=path, fmt="parquet")
            assert os.path.exists(result_path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_supported_formats_tuple(self):
        assert "json" in _SUPPORTED_FORMATS
        assert "csv" in _SUPPORTED_FORMATS
        assert "parquet" in _SUPPORTED_FORMATS


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------


class TestMain:
    """Tests for the main() entry point."""

    def test_version_flag(self, capsys):
        code = main(["--version"])
        assert code == 0
        out = capsys.readouterr().out
        assert "qufin" in out

    def test_no_subcommand_returns_1(self):
        code = main([])
        assert code == 1

    def test_optimize_json_output(self, capsys):
        code = main([
            "optimize", "--universe", "sp500", "--method", "qaoa",
            "--cardinality", "20",
        ])
        assert code == 0
        out = capsys.readouterr().out
        loaded = json.loads(out)
        assert loaded["command"] == "optimize"

    def test_price_json_output(self, capsys):
        code = main([
            "price", "--type", "european", "--s", "100", "--k", "105",
            "--method", "iqae",
        ])
        assert code == 0
        out = capsys.readouterr().out
        loaded = json.loads(out)
        assert loaded["data"]["solver"] == "quantum"

    def test_risk_json_output(self, capsys):
        code = main([
            "risk", "--portfolio", "port.csv", "--method", "quantum_var",
        ])
        assert code == 0

    def test_benchmark_json_output(self, capsys):
        code = main([
            "benchmark", "--problem", "15", "--algorithms", "qaoa,vqe",
        ])
        assert code == 0


# ------------------------------------------------------------------
# Shell completion
# ------------------------------------------------------------------


class TestShellCompletion:
    """Tests for shell completion generation."""

    def test_bash_completion(self):
        script = install_completion("bash")
        assert "bash" in script.lower()

    def test_zsh_completion(self):
        script = install_completion("zsh")
        assert "zsh" in script.lower()

    def test_fish_completion(self):
        script = install_completion("fish")
        assert "fish" in script.lower()

    def test_unsupported_shell(self):
        text = install_completion("powershell")
        assert "Unsupported" in text
