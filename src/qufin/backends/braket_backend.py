"""AWS Braket backend adapter (IonQ, Rigetti, IQM, QuEra).

Provides a qufin Backend interface wrapping Amazon Braket devices
and simulators, with target-specific metadata, hybrid job support,
and cost estimation.

Requires: pip install qufin[braket]
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qufin.backends.base import Backend, CircuitResult

# ---------------------------------------------------------------------------
# Device target dataclasses
# ---------------------------------------------------------------------------


class Topology(enum.Enum):
    """QPU connectivity topology."""

    ALL_TO_ALL = "all_to_all"
    GRID = "grid"
    HEAVY_HEX = "heavy_hex"


@dataclass(frozen=True)
class IonQTarget:
    """IonQ trapped-ion device target.

    Attributes
    ----------
    name : str
        Human-readable device name.
    arn : str
        AWS device ARN.
    topology : Topology
        Qubit connectivity (all-to-all for trapped ion).
    max_qubits : int
        Maximum number of qubits supported.
    swap_overhead : float
        SWAP overhead factor for QAOA-style circuits (1.0 = no overhead).
    """

    name: str
    arn: str
    topology: Topology = Topology.ALL_TO_ALL
    max_qubits: int = 25
    swap_overhead: float = 1.0  # no SWAP overhead for all-to-all


IONQ_ARIA = IonQTarget(
    name="IonQ Aria",
    arn="arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1",
    max_qubits=25,
)

IONQ_FORTE = IonQTarget(
    name="IonQ Forte",
    arn="arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1",
    max_qubits=32,
)


@dataclass(frozen=True)
class RigettiTarget:
    """Rigetti superconducting device target.

    Attributes
    ----------
    name : str
        Human-readable device name.
    arn : str
        AWS device ARN.
    topology : Topology
        Qubit connectivity (grid for superconducting).
    max_qubits : int
        Maximum number of qubits supported.
    swap_overhead : float
        Estimated SWAP overhead factor for QAOA circuits on grid topology.
    """

    name: str
    arn: str
    topology: Topology = Topology.GRID
    max_qubits: int = 84
    swap_overhead: float = 3.0  # grid needs routing


RIGETTI_ANKAA = RigettiTarget(
    name="Rigetti Ankaa-2",
    arn="arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-2",
    max_qubits=84,
)


@dataclass(frozen=True)
class IQMTarget:
    """IQM superconducting device target."""

    name: str
    arn: str
    topology: Topology = Topology.GRID
    max_qubits: int = 20
    swap_overhead: float = 2.5


IQM_GARNET = IQMTarget(
    name="IQM Garnet",
    arn="arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet",
    max_qubits=20,
)

# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

# Per-shot pricing (USD) and per-task fixed cost.
# Source: AWS Braket pricing page (approximate, subject to change).

_DeviceTarget = IonQTarget | RigettiTarget | IQMTarget


@dataclass(frozen=True)
class _PriceEntry:
    per_task_usd: float
    per_shot_usd: float


_PRICE_TABLE: dict[str, _PriceEntry] = {
    IONQ_ARIA.arn: _PriceEntry(per_task_usd=0.30, per_shot_usd=0.01),
    IONQ_FORTE.arn: _PriceEntry(per_task_usd=0.30, per_shot_usd=0.01),
    RIGETTI_ANKAA.arn: _PriceEntry(per_task_usd=0.30, per_shot_usd=0.00035),
    IQM_GARNET.arn: _PriceEntry(per_task_usd=0.30, per_shot_usd=0.00045),
}


@dataclass(frozen=True)
class CostEstimate:
    """Estimated cost for running a circuit.

    Attributes
    ----------
    device_name : str
        Human-readable device name.
    device_arn : str
        AWS device ARN.
    shots : int
        Number of shots requested.
    per_task_usd : float
        Fixed cost per task submission.
    per_shot_usd : float
        Cost per shot.
    total_usd : float
        Total estimated cost.
    """

    device_name: str
    device_arn: str
    shots: int
    per_task_usd: float
    per_shot_usd: float
    total_usd: float


def estimate_cost(
    target: _DeviceTarget,
    shots: int = 1024,
    n_tasks: int = 1,
) -> CostEstimate:
    """Estimate the cost of running on a Braket QPU.

    Parameters
    ----------
    target : IonQTarget | RigettiTarget | IQMTarget
        Device target with ARN.
    shots : int
        Number of shots per task.
    n_tasks : int
        Number of task submissions.

    Returns
    -------
    CostEstimate
        Breakdown of estimated costs.

    Raises
    ------
    ValueError
        If the target ARN is not in the price table.
    """
    entry = _PRICE_TABLE.get(target.arn)
    if entry is None:
        raise ValueError(
            f"No pricing data for {target.arn}. "
            f"Known devices: {list(_PRICE_TABLE.keys())}"
        )
    total = n_tasks * (entry.per_task_usd + shots * entry.per_shot_usd)
    return CostEstimate(
        device_name=target.name,
        device_arn=target.arn,
        shots=shots,
        per_task_usd=entry.per_task_usd,
        per_shot_usd=entry.per_shot_usd,
        total_usd=round(total, 6),
    )


# ---------------------------------------------------------------------------
# Circuit analysis helpers
# ---------------------------------------------------------------------------


def analyze_swap_overhead(
    target: _DeviceTarget,
    n_qubits: int,
    n_two_qubit_gates: int,
) -> dict[str, Any]:
    """Estimate SWAP overhead for a circuit on a given target.

    For all-to-all connectivity (trapped ion), no SWAPs are needed.
    For grid/heavy-hex topologies, overhead scales with connectivity distance.

    Parameters
    ----------
    target : IonQTarget | RigettiTarget | IQMTarget
        Device target.
    n_qubits : int
        Number of logical qubits.
    n_two_qubit_gates : int
        Number of two-qubit gates in the logical circuit.

    Returns
    -------
    dict
        Analysis with keys: estimated_swaps, total_two_qubit_gates,
        overhead_factor, topology.
    """
    if n_qubits > target.max_qubits:
        raise ValueError(
            f"Circuit requires {n_qubits} qubits but {target.name} "
            f"supports at most {target.max_qubits}"
        )

    if target.topology == Topology.ALL_TO_ALL:
        estimated_swaps = 0
        total_2q = n_two_qubit_gates
    else:
        # Heuristic: for grid topology, average routing distance ~ sqrt(n_qubits)
        avg_distance = math.sqrt(n_qubits)
        estimated_swaps = int(n_two_qubit_gates * (avg_distance - 1) * 0.5)
        # Each SWAP = 3 CNOTs
        total_2q = n_two_qubit_gates + 3 * estimated_swaps

    overhead = total_2q / max(n_two_qubit_gates, 1)
    return {
        "estimated_swaps": estimated_swaps,
        "total_two_qubit_gates": total_2q,
        "overhead_factor": round(overhead, 2),
        "topology": target.topology.value,
    }


# ---------------------------------------------------------------------------
# Hybrid job support
# ---------------------------------------------------------------------------


@dataclass
class HybridJobHandle:
    """Handle for a submitted Braket hybrid job.

    Attributes
    ----------
    job_arn : str
        ARN of the submitted hybrid job.
    status : str
        Current job status.
    metadata : dict
        Additional job metadata.
    """

    job_arn: str
    status: str = "QUEUED"
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main backend class
# ---------------------------------------------------------------------------


class BraketBackend(Backend):
    """Amazon Braket backend.

    Supports local simulator, managed QPU devices, hybrid jobs,
    and cost estimation.

    Parameters
    ----------
    device_arn : str or None
        AWS device ARN. None uses local simulator.
    s3_bucket : str or None
        S3 bucket for results (required for QPU).
    s3_prefix : str
        S3 key prefix for results.
    """

    def __init__(
        self,
        device_arn: str | None = None,
        s3_bucket: str | None = None,
        s3_prefix: str = "qufin-results",
    ) -> None:
        try:
            from braket.devices import LocalSimulator
        except ImportError as e:
            raise ImportError(
                "Amazon Braket SDK is required. "
                "Install with: pip install qufin[braket]"
            ) from e

        self._device_arn = device_arn
        self._s3_bucket = s3_bucket
        self._s3_prefix = s3_prefix

        if device_arn is None:
            from braket.devices import LocalSimulator

            self._device = LocalSimulator()
            self._is_local = True
        else:
            from braket.aws import AwsDevice

            self._device = AwsDevice(device_arn)
            self._is_local = False

    @property
    def backend_id(self) -> str:
        if self._device_arn:
            return f"braket:{self._device_arn.split('/')[-1]}"
        return "braket:local"

    def run(self, circuit: Any, shots: int = 1024) -> CircuitResult:
        """Execute a circuit and return measurement results."""
        if hasattr(circuit, "num_qubits"):
            return self._run_qiskit_circuit(circuit, shots)

        # Native Braket circuit
        from braket.circuits import Circuit as BraketCircuit

        if isinstance(circuit, BraketCircuit):
            return self._run_braket_circuit(circuit, shots)

        return CircuitResult(counts={}, shots=shots, backend_id=self.backend_id)

    def _run_braket_circuit(self, circuit: Any, shots: int) -> CircuitResult:
        """Run a native Braket circuit."""
        if self._is_local:
            task = self._device.run(circuit, shots=shots)
        else:
            s3_loc = (self._s3_bucket, self._s3_prefix)
            task = self._device.run(circuit, s3_loc, shots=shots)

        result = task.result()
        counts = {k: int(v) for k, v in result.measurement_counts.items()}
        return CircuitResult(
            counts=counts, shots=shots, backend_id=self.backend_id
        )

    def _run_qiskit_circuit(self, circuit: Any, shots: int) -> CircuitResult:
        """Run a Qiskit circuit by converting to Braket via OpenQASM."""
        try:
            from braket.circuits import Circuit as BraketCircuit
            from qiskit import qasm2, transpile

            transpiled = transpile(
                circuit,
                basis_gates=["cx", "u3", "id", "measure"],
                optimization_level=0,
            )
            qasm_str = qasm2.dumps(transpiled)
            braket_circuit = BraketCircuit.from_ir(
                qasm_str, ir_type="OPENQASM"
            )
            return self._run_braket_circuit(braket_circuit, shots)
        except Exception:
            return CircuitResult(
                counts={}, shots=shots, backend_id=self.backend_id
            )

    def statevector(self, circuit: Any) -> NDArray[np.complex128]:
        """Return the statevector (local simulator only)."""
        if not self._is_local:
            raise RuntimeError("Statevector only available on local simulator")

        from braket.devices import LocalSimulator

        sv_device = LocalSimulator("default")

        if hasattr(circuit, "num_qubits"):
            try:
                from braket.circuits import Circuit as BraketCircuit
                from qiskit import qasm2, transpile

                transpiled = transpile(
                    circuit,
                    basis_gates=["cx", "u3", "id"],
                    optimization_level=0,
                )
                qasm_str = qasm2.dumps(transpiled)
                braket_circuit = BraketCircuit.from_ir(
                    qasm_str, ir_type="OPENQASM"
                )
                braket_circuit.state_vector()
                task = sv_device.run(braket_circuit, shots=0)
                result = task.result()
                return np.array(
                    result.result_types[0].value, dtype=np.complex128
                )
            except Exception:
                pass

        raise ValueError("Cannot compute statevector")

    def is_simulator(self) -> bool:
        return self._is_local

    # -----------------------------------------------------------------------
    # Hybrid job support
    # -----------------------------------------------------------------------

    def submit_hybrid_job(
        self,
        algorithm_script: str,
        *,
        entry_point: str = "algorithm_script:main",
        hyperparameters: dict[str, str] | None = None,
        instance_type: str = "ml.m5.large",
        wait_until_complete: bool = False,
    ) -> HybridJobHandle:
        """Submit a Braket hybrid job for long-running optimizations.

        Parameters
        ----------
        algorithm_script : str
            Path to the Python script containing the algorithm.
        entry_point : str
            Module and function entry point (default: ``algorithm_script:main``).
        hyperparameters : dict or None
            Hyperparameters passed to the algorithm script.
        instance_type : str
            EC2 instance type for the classical co-processor.
        wait_until_complete : bool
            If True, block until the job completes.

        Returns
        -------
        HybridJobHandle
            Handle with job ARN and initial status.

        Raises
        ------
        RuntimeError
            If using local simulator (hybrid jobs require QPU/on-demand).
        ImportError
            If braket SDK hybrid job module is unavailable.
        """
        if self._is_local:
            raise RuntimeError(
                "Hybrid jobs require a QPU or on-demand simulator device, "
                "not the local simulator."
            )

        from braket.aws import AwsQuantumJob

        job_params: dict[str, Any] = {
            "device": self._device_arn,
            "source_module": algorithm_script,
            "entry_point": entry_point,
            "instance_config": {"instanceType": instance_type},
        }
        if hyperparameters:
            job_params["hyperparameters"] = hyperparameters

        if self._s3_bucket:
            job_params["output_data_config"] = {
                "s3Path": f"s3://{self._s3_bucket}/{self._s3_prefix}/jobs",
            }

        job = AwsQuantumJob.create(**job_params)

        if wait_until_complete:
            job.result()  # blocks until done

        state = job.state()
        return HybridJobHandle(
            job_arn=job.arn,
            status=state,
            metadata={"instance_type": instance_type},
        )

    def poll_job_status(self, job_arn: str) -> HybridJobHandle:
        """Poll the status of a previously submitted hybrid job.

        Parameters
        ----------
        job_arn : str
            ARN of the hybrid job.

        Returns
        -------
        HybridJobHandle
            Updated handle with current status.
        """
        from braket.aws import AwsQuantumJob

        job = AwsQuantumJob(arn=job_arn)
        state = job.state()
        metadata: dict[str, Any] = {}
        if state == "COMPLETED":
            try:
                metadata["result"] = job.result()
            except Exception:
                metadata["result_error"] = "Could not retrieve result"

        return HybridJobHandle(
            job_arn=job_arn,
            status=state,
            metadata=metadata,
        )

    def get_job_result(self, job_arn: str) -> dict[str, Any]:
        """Retrieve the result of a completed hybrid job.

        Parameters
        ----------
        job_arn : str
            ARN of the hybrid job.

        Returns
        -------
        dict
            Job result dictionary. Empty dict if job is not completed.

        Raises
        ------
        RuntimeError
            If the job has failed.
        """
        from braket.aws import AwsQuantumJob

        job = AwsQuantumJob(arn=job_arn)
        state = job.state()
        if state == "FAILED":
            raise RuntimeError(f"Hybrid job {job_arn} has FAILED.")
        if state != "COMPLETED":
            return {"status": state, "message": "Job not yet completed."}
        return {"status": "COMPLETED", "result": job.result()}

    # -----------------------------------------------------------------------
    # Cost estimation (convenience instance method)
    # -----------------------------------------------------------------------

    def estimate_cost(self, shots: int = 1024, n_tasks: int = 1) -> CostEstimate:
        """Estimate cost for this backend's device.

        Parameters
        ----------
        shots : int
            Number of shots per task.
        n_tasks : int
            Number of task submissions.

        Returns
        -------
        CostEstimate

        Raises
        ------
        ValueError
            If device ARN is not in the price table or backend is local.
        """
        if self._is_local or self._device_arn is None:
            raise ValueError("Cost estimation not available for local simulator.")
        # Build a minimal target just for the lookup
        _target = IonQTarget(name=self.backend_id, arn=self._device_arn)
        return estimate_cost(_target, shots=shots, n_tasks=n_tasks)
