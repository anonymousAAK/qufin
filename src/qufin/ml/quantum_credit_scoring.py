"""Projected quantum kernel credit scoring with fairness analysis.

Implements a projected quantum kernel for credit scoring tasks,
with benchmark framework comparing quantum kernel SVM against
classical baselines (RBF-SVM, XGBoost, logistic regression).

Includes fairness metrics: statistical parity and equal opportunity
across protected attributes.

References
----------
Huang et al., Nature Communications 12, 2631 (2021) — projected quantum kernels.
Havlicek et al., Nature 567, 209-212 (2019) — quantum kernel methods.
"""

from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from qufin.backends.base import Backend
from qufin.ml.kernels import ZZFeatureMap, quantum_kernel_matrix

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    from sklearn.linear_model import LogisticRegression  # noqa: F401
    from sklearn.metrics import (  # noqa: F401
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: F401
    from sklearn.preprocessing import StandardScaler  # noqa: F401
    from sklearn.svm import SVC  # noqa: F401

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    from xgboost import XGBClassifier  # noqa: F401

    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

try:
    from scipy.stats import ttest_rel  # noqa: F401

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _require_sklearn() -> None:
    """Raise if scikit-learn is not installed."""
    if not _HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is required for credit scoring benchmarks. "
            "Install with: pip install scikit-learn"
        )


# ---------------------------------------------------------------------------
# Feature map variants
# ---------------------------------------------------------------------------


class FeatureMapType(Enum):
    """Supported quantum feature map types."""

    ZZ = "zz"
    IQP = "iqp"


class IQPFeatureMap:
    """IQP (Instantaneous Quantum Polynomial) feature map circuit builder.

    Encodes classical data using diagonal unitals with Hadamard layers,
    producing circuits in the IQP class.
    """

    @staticmethod
    def build_circuit(
        x: NDArray[np.float64], n_qubits: int, reps: int = 2
    ) -> Any:
        """Build an IQP feature map circuit for data vector *x*.

        Parameters
        ----------
        x : array of shape (n_qubits,)
            Classical feature vector.
        n_qubits : int
            Number of qubits.
        reps : int
            Repetitions of the IQP layer.

        Returns
        -------
        QuantumCircuit
        """
        from qiskit.circuit import QuantumCircuit

        qc = QuantumCircuit(n_qubits)

        for _ in range(reps):
            # Hadamard layer
            for i in range(n_qubits):
                qc.h(i)

            # Diagonal unitary: single-qubit RZ
            for i in range(n_qubits):
                qc.rz(x[i], i)

            # Two-qubit diagonal: CNOT-RZ-CNOT (encodes product terms)
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    phi = x[i] * x[j]
                    qc.cx(i, j)
                    qc.rz(phi, j)
                    qc.cx(i, j)

        return qc


# ---------------------------------------------------------------------------
# Projected quantum kernel
# ---------------------------------------------------------------------------


@dataclass
class ProjectedKernelConfig:
    """Configuration for the projected quantum kernel.

    Parameters
    ----------
    n_qubits : int
        Number of qubits for the feature map.
    feature_map : FeatureMapType
        Which feature map circuit to use.
    reps : int
        Number of feature map repetitions.
    gamma_projection : float
        Bandwidth parameter for the classical RBF projection step.
    """

    n_qubits: int = 4
    feature_map: FeatureMapType = FeatureMapType.ZZ
    reps: int = 2
    gamma_projection: float = 1.0


def _build_feature_map_circuit(
    x: NDArray[np.float64],
    n_qubits: int,
    feature_map: FeatureMapType,
    reps: int,
) -> Any:
    """Build a feature map circuit for the given type."""
    if feature_map == FeatureMapType.ZZ:
        return ZZFeatureMap.build_circuit(x, n_qubits, reps)
    elif feature_map == FeatureMapType.IQP:
        return IQPFeatureMap.build_circuit(x, n_qubits, reps)
    else:
        raise ValueError(f"Unknown feature map type: {feature_map}")


def projected_quantum_state(
    x: NDArray[np.float64],
    backend: Backend,
    config: ProjectedKernelConfig,
) -> NDArray[np.float64]:
    """Compute the projected quantum state for a single data point.

    Projects the quantum state onto single-qubit reduced density matrices
    and extracts Bloch vector components, producing a classical feature
    vector of dimension 3 * n_qubits.

    Parameters
    ----------
    x : array of shape (n_qubits,)
        Input feature vector.
    backend : Backend
        Quantum backend for statevector simulation.
    config : ProjectedKernelConfig
        Kernel configuration.

    Returns
    -------
    NDArray of shape (3 * n_qubits,)
        Projected classical feature vector (Bloch vector components).
    """
    circ = _build_feature_map_circuit(x, config.n_qubits, config.feature_map, config.reps)
    sv = backend.statevector(circ)

    n_q = config.n_qubits
    n_states = 2**n_q
    projected = np.zeros(3 * n_q, dtype=np.float64)

    # For each qubit, compute reduced density matrix and extract Bloch vector
    for q in range(n_q):
        # Trace out all qubits except q to get 2x2 reduced density matrix
        rho = np.zeros((2, 2), dtype=np.complex128)
        for i in range(n_states):
            for j in range(n_states):
                # Check if qubits other than q are the same
                mask = ~(1 << q) & ((1 << n_q) - 1)
                if (i & mask) == (j & mask):
                    bit_i = (i >> q) & 1
                    bit_j = (j >> q) & 1
                    rho[bit_i, bit_j] += sv[i] * np.conj(sv[j])

        # Bloch vector: (Tr[rho*X], Tr[rho*Y], Tr[rho*Z])
        projected[3 * q] = 2.0 * np.real(rho[0, 1])       # <X>
        projected[3 * q + 1] = 2.0 * np.imag(rho[1, 0])   # <Y>
        projected[3 * q + 2] = np.real(rho[0, 0] - rho[1, 1])  # <Z>

    return projected


def projected_kernel_matrix(
    X: NDArray[np.float64],
    backend: Backend,
    config: ProjectedKernelConfig,
) -> NDArray[np.float64]:
    """Compute the projected quantum kernel matrix.

    Projects each data point through the quantum feature map, then
    computes an RBF kernel on the projected (classical) features.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Input data. Features are truncated/padded to n_qubits.
    backend : Backend
        Quantum backend.
    config : ProjectedKernelConfig
        Kernel configuration.

    Returns
    -------
    K : array of shape (n_samples, n_samples)
        Kernel matrix.
    """
    n_samples = X.shape[0]

    # Truncate or pad features to n_qubits
    X_q = _prepare_features(X, config.n_qubits)

    # Project all data points
    projections = np.zeros((n_samples, 3 * config.n_qubits), dtype=np.float64)
    for i in range(n_samples):
        projections[i] = projected_quantum_state(X_q[i], backend, config)

    # Compute RBF kernel on projected features
    K = np.zeros((n_samples, n_samples), dtype=np.float64)
    for i in range(n_samples):
        for j in range(i, n_samples):
            diff = projections[i] - projections[j]
            k_val = np.exp(-config.gamma_projection * np.dot(diff, diff))
            K[i, j] = k_val
            K[j, i] = k_val

    return K


def _prepare_features(
    X: NDArray[np.float64], n_qubits: int
) -> NDArray[np.float64]:
    """Truncate or pad features to match n_qubits, then scale to [0, 2*pi]."""
    n_samples, n_features = X.shape
    X_q = np.zeros((n_samples, n_qubits), dtype=np.float64)
    k = min(n_features, n_qubits)
    X_q[:, :k] = X[:, :k]

    # Scale each feature to [0, 2*pi]
    for col in range(k):
        col_min = X_q[:, col].min()
        col_max = X_q[:, col].max()
        if col_max > col_min:
            X_q[:, col] = (X_q[:, col] - col_min) / (col_max - col_min) * 2 * np.pi
        else:
            X_q[:, col] = 0.0

    return X_q


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------


@dataclass
class CreditDataset:
    """Container for a credit scoring dataset.

    Attributes
    ----------
    X : NDArray
        Feature matrix, shape (n_samples, n_features).
    y : NDArray
        Labels (0 = good credit, 1 = bad credit), shape (n_samples,).
    protected_attr : NDArray | None
        Protected attribute vector for fairness analysis.
    feature_names : list[str]
        Names of features.
    name : str
        Dataset name.
    """

    X: NDArray[np.float64]
    y: NDArray[np.int64]
    protected_attr: NDArray[np.int64] | None = None
    feature_names: list[str] = field(default_factory=list)
    name: str = ""


def load_german_credit(
    protected_attribute: str = "age",
) -> CreditDataset:
    """Load the UCI German Credit dataset.

    Attempts to load from sklearn.datasets or falls back to generating
    a synthetic dataset with the same statistical properties.

    Parameters
    ----------
    protected_attribute : str
        Which attribute to use for fairness analysis ("age" or "sex").

    Returns
    -------
    CreditDataset
    """
    _require_sklearn()
    from sklearn.datasets import fetch_openml

    try:
        data = fetch_openml(data_id=31, as_frame=True, parser="auto")
        df = data.frame
        y = (df["class"].astype(int) == 2).astype(np.int64).values

        # Extract protected attribute before dropping
        if protected_attribute == "age":
            protected = (df["personal_status"].str.contains("A91|A93|A94")).astype(
                np.int64
            ).values
        else:
            protected = (df["personal_status"].str.contains("A92|A95")).astype(
                np.int64
            ).values

        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if "class" in numeric_cols:
            numeric_cols.remove("class")
        X = df[numeric_cols].values.astype(np.float64)

        return CreditDataset(
            X=X,
            y=y,
            protected_attr=protected,
            feature_names=numeric_cols,
            name="german_credit",
        )
    except Exception:
        return _synthetic_credit_dataset(
            n_samples=1000,
            n_features=20,
            name="german_credit_synthetic",
        )


def load_taiwan_credit() -> CreditDataset:
    """Load the Taiwan Credit default dataset.

    Attempts to load from OpenML or falls back to a synthetic dataset
    with similar statistical properties.

    Returns
    -------
    CreditDataset
    """
    _require_sklearn()
    from sklearn.datasets import fetch_openml

    try:
        data = fetch_openml(data_id=42477, as_frame=True, parser="auto")
        df = data.frame
        target_col = df.columns[-1]
        y = df[target_col].astype(np.int64).values

        # Sex column (2=female, 1=male) as protected attribute
        protected = None
        if "SEX" in df.columns:
            protected = (df["SEX"].astype(int) == 2).astype(np.int64).values
        elif "X2" in df.columns:
            protected = (df["X2"].astype(int) == 2).astype(np.int64).values

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        X = df[numeric_cols].values.astype(np.float64)

        return CreditDataset(
            X=X,
            y=y,
            protected_attr=protected,
            feature_names=numeric_cols,
            name="taiwan_credit",
        )
    except Exception:
        return _synthetic_credit_dataset(
            n_samples=3000,
            n_features=23,
            name="taiwan_credit_synthetic",
        )


def _synthetic_credit_dataset(
    n_samples: int = 1000,
    n_features: int = 20,
    name: str = "synthetic_credit",
    random_state: int = 42,
) -> CreditDataset:
    """Generate a synthetic credit dataset for testing."""
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n_samples, n_features))
    # Create a separable problem: weighted sum with noise
    weights = rng.standard_normal(n_features)
    score = X @ weights + rng.standard_normal(n_samples) * 0.5
    y = (score > np.median(score)).astype(np.int64)
    protected = rng.integers(0, 2, size=n_samples).astype(np.int64)

    feature_names = [f"feature_{i}" for i in range(n_features)]
    return CreditDataset(
        X=X,
        y=y,
        protected_attr=protected,
        feature_names=feature_names,
        name=name,
    )


# ---------------------------------------------------------------------------
# Benchmark framework
# ---------------------------------------------------------------------------


@dataclass
class ClassifierMetrics:
    """Evaluation metrics for a single classifier.

    Attributes
    ----------
    accuracy : float
    auc_roc : float
    precision : float
    recall : float
    f1 : float
    cv_scores : NDArray[np.float64] | None
        Per-fold accuracy scores from cross-validation.
    """

    accuracy: float = 0.0
    auc_roc: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    cv_scores: NDArray[np.float64] | None = None


@dataclass
class BenchmarkResult:
    """Results from the credit scoring benchmark.

    Attributes
    ----------
    classifier_name : str
    metrics : ClassifierMetrics
    train_time_seconds : float
    n_samples : int
    n_features : int
    """

    classifier_name: str = ""
    metrics: ClassifierMetrics = field(default_factory=ClassifierMetrics)
    train_time_seconds: float = 0.0
    n_samples: int = 0
    n_features: int = 0


@dataclass
class PairedTestResult:
    """Result of a paired t-test between two classifiers.

    Attributes
    ----------
    classifier_a : str
    classifier_b : str
    t_statistic : float
    p_value : float
    significant : bool
        Whether the difference is significant at alpha=0.05.
    """

    classifier_a: str = ""
    classifier_b: str = ""
    t_statistic: float = 0.0
    p_value: float = 1.0
    significant: bool = False


def _compute_metrics(
    y_true: NDArray, y_pred: NDArray, y_prob: NDArray | None = None
) -> ClassifierMetrics:
    """Compute classification metrics."""
    _require_sklearn()
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    auc = 0.0
    if y_prob is not None:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auc = 0.0

    return ClassifierMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        auc_roc=auc,
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
    )


def evaluate_quantum_kernel_svm(
    X_train: NDArray[np.float64],
    y_train: NDArray,
    X_test: NDArray[np.float64],
    y_test: NDArray,
    backend: Backend,
    config: ProjectedKernelConfig | None = None,
    use_projected: bool = True,
    C: float = 1.0,
) -> BenchmarkResult:
    """Evaluate quantum kernel SVM on credit scoring data.

    Parameters
    ----------
    X_train, y_train : Training data.
    X_test, y_test : Test data.
    backend : Quantum backend.
    config : Kernel configuration. Defaults to 4-qubit ZZ.
    use_projected : If True, use projected kernel; else full quantum kernel.
    C : SVM regularization.

    Returns
    -------
    BenchmarkResult
    """
    _require_sklearn()
    import time

    from sklearn.svm import SVC

    if config is None:
        config = ProjectedKernelConfig()

    start = time.perf_counter()

    if use_projected:
        X_train_q = _prepare_features(X_train, config.n_qubits)
        X_test_q = _prepare_features(X_test, config.n_qubits)

        # Compute projected kernel matrices
        K_train = projected_kernel_matrix(X_train_q, backend, config)

        # Test kernel: K_test[i,j] = kernel(test_i, train_j)
        n_test = X_test_q.shape[0]
        n_train = X_train_q.shape[0]
        proj_train = np.zeros((n_train, 3 * config.n_qubits), dtype=np.float64)
        for i in range(n_train):
            proj_train[i] = projected_quantum_state(X_train_q[i], backend, config)

        proj_test = np.zeros((n_test, 3 * config.n_qubits), dtype=np.float64)
        for i in range(n_test):
            proj_test[i] = projected_quantum_state(X_test_q[i], backend, config)

        K_test = np.zeros((n_test, n_train), dtype=np.float64)
        for i in range(n_test):
            for j in range(n_train):
                diff = proj_test[i] - proj_train[j]
                K_test[i, j] = np.exp(-config.gamma_projection * np.dot(diff, diff))
    else:
        X_train_q = _prepare_features(X_train, config.n_qubits)
        X_test_q = _prepare_features(X_test, config.n_qubits)
        K_train = quantum_kernel_matrix(X_train_q, config.n_qubits, backend, config.reps)

        from qufin.ml.kernels import quantum_kernel

        n_test = X_test_q.shape[0]
        n_train = X_train_q.shape[0]
        K_test = np.zeros((n_test, n_train), dtype=np.float64)
        for i in range(n_test):
            for j in range(n_train):
                K_test[i, j] = quantum_kernel(
                    X_test_q[i], X_train_q[j], config.n_qubits, backend, config.reps
                )

    svc = SVC(kernel="precomputed", C=C, probability=True)
    svc.fit(K_train, y_train)
    y_pred = svc.predict(K_test)
    try:
        y_prob = svc.predict_proba(K_test)[:, 1]
    except Exception:
        y_prob = None

    elapsed = time.perf_counter() - start
    metrics = _compute_metrics(y_test, y_pred, y_prob)

    return BenchmarkResult(
        classifier_name="quantum_kernel_svm",
        metrics=metrics,
        train_time_seconds=elapsed,
        n_samples=len(y_train) + len(y_test),
        n_features=X_train.shape[1],
    )


def evaluate_classical_baseline(
    X_train: NDArray[np.float64],
    y_train: NDArray,
    X_test: NDArray[np.float64],
    y_test: NDArray,
    classifier: Literal["rbf_svm", "xgboost", "logistic_regression"] = "rbf_svm",
    C: float = 1.0,
) -> BenchmarkResult:
    """Evaluate a classical baseline classifier.

    Parameters
    ----------
    X_train, y_train, X_test, y_test : Data splits.
    classifier : Which classical classifier.
    C : Regularization parameter.

    Returns
    -------
    BenchmarkResult
    """
    _require_sklearn()
    import time

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    start = time.perf_counter()

    if classifier == "rbf_svm":
        from sklearn.svm import SVC

        model = SVC(kernel="rbf", C=C, probability=True)
    elif classifier == "xgboost":
        if not _HAS_XGBOOST:
            raise ImportError(
                "XGBoost is required. Install with: pip install xgboost"
            )
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=100, max_depth=3, use_label_encoder=False,
            eval_metric="logloss", verbosity=0,
        )
    elif classifier == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    y_prob = None
    if hasattr(model, "predict_proba"):
        with contextlib.suppress(Exception):
            y_prob = model.predict_proba(X_test_s)[:, 1]

    elapsed = time.perf_counter() - start
    metrics = _compute_metrics(y_test, y_pred, y_prob)

    return BenchmarkResult(
        classifier_name=classifier,
        metrics=metrics,
        train_time_seconds=elapsed,
        n_samples=len(y_train) + len(y_test),
        n_features=X_train.shape[1],
    )


def cross_validate_classifier(
    X: NDArray[np.float64],
    y: NDArray,
    classifier: Literal["rbf_svm", "xgboost", "logistic_regression"],
    k_folds: int = 5,
    C: float = 1.0,
    random_state: int = 42,
) -> NDArray[np.float64]:
    """Run k-fold cross-validation for a classical classifier.

    Returns per-fold accuracy scores.
    """
    _require_sklearn()
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if classifier == "rbf_svm":
        from sklearn.svm import SVC

        model = SVC(kernel="rbf", C=C)
    elif classifier == "xgboost":
        if not _HAS_XGBOOST:
            raise ImportError("XGBoost is required.")
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=100, max_depth=3, use_label_encoder=False,
            eval_metric="logloss", verbosity=0,
        )
    elif classifier == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
    cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    return scores.astype(np.float64)


def paired_t_test(
    scores_a: NDArray[np.float64],
    scores_b: NDArray[np.float64],
    name_a: str = "A",
    name_b: str = "B",
    alpha: float = 0.05,
) -> PairedTestResult:
    """Perform a paired t-test on cross-validation scores.

    Parameters
    ----------
    scores_a, scores_b : Per-fold scores for two classifiers.
    name_a, name_b : Classifier names.
    alpha : Significance level.

    Returns
    -------
    PairedTestResult
    """
    if not _HAS_SCIPY:
        raise ImportError("scipy is required for statistical tests.")
    from scipy.stats import ttest_rel

    if len(scores_a) != len(scores_b):
        raise ValueError("Score arrays must have the same length.")

    t_stat, p_val = ttest_rel(scores_a, scores_b)

    return PairedTestResult(
        classifier_a=name_a,
        classifier_b=name_b,
        t_statistic=float(t_stat),
        p_value=float(p_val),
        significant=float(p_val) < alpha,
    )


# ---------------------------------------------------------------------------
# Fairness analysis
# ---------------------------------------------------------------------------


@dataclass
class FairnessMetrics:
    """Fairness metrics for a classifier's predictions.

    Attributes
    ----------
    statistical_parity_difference : float
        Difference in positive prediction rates between groups.
        |P(Y_hat=1|A=0) - P(Y_hat=1|A=1)|.
    equal_opportunity_difference : float
        Difference in true positive rates between groups.
        |P(Y_hat=1|Y=1,A=0) - P(Y_hat=1|Y=1,A=1)|.
    positive_rate_group_0 : float
        P(Y_hat=1 | A=0).
    positive_rate_group_1 : float
        P(Y_hat=1 | A=1).
    tpr_group_0 : float
        True positive rate for group 0.
    tpr_group_1 : float
        True positive rate for group 1.
    """

    statistical_parity_difference: float = 0.0
    equal_opportunity_difference: float = 0.0
    positive_rate_group_0: float = 0.0
    positive_rate_group_1: float = 0.0
    tpr_group_0: float = 0.0
    tpr_group_1: float = 0.0


def compute_fairness_metrics(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    protected_attr: NDArray[np.int64],
) -> FairnessMetrics:
    """Compute fairness metrics across a binary protected attribute.

    Parameters
    ----------
    y_true : True labels, shape (n,).
    y_pred : Predicted labels, shape (n,).
    protected_attr : Binary protected attribute (0 or 1), shape (n,).

    Returns
    -------
    FairnessMetrics
    """
    mask_0 = protected_attr == 0
    mask_1 = protected_attr == 1

    # Statistical parity: P(Y_hat=1 | A=group)
    n_0 = mask_0.sum()
    n_1 = mask_1.sum()

    pos_rate_0 = float(y_pred[mask_0].sum() / n_0) if n_0 > 0 else 0.0
    pos_rate_1 = float(y_pred[mask_1].sum() / n_1) if n_1 > 0 else 0.0

    # Equal opportunity: P(Y_hat=1 | Y=1, A=group)
    pos_true_0 = mask_0 & (y_true == 1)
    pos_true_1 = mask_1 & (y_true == 1)

    n_pos_0 = pos_true_0.sum()
    n_pos_1 = pos_true_1.sum()

    tpr_0 = float(y_pred[pos_true_0].sum() / n_pos_0) if n_pos_0 > 0 else 0.0
    tpr_1 = float(y_pred[pos_true_1].sum() / n_pos_1) if n_pos_1 > 0 else 0.0

    return FairnessMetrics(
        statistical_parity_difference=abs(pos_rate_0 - pos_rate_1),
        equal_opportunity_difference=abs(tpr_0 - tpr_1),
        positive_rate_group_0=pos_rate_0,
        positive_rate_group_1=pos_rate_1,
        tpr_group_0=tpr_0,
        tpr_group_1=tpr_1,
    )


def run_full_benchmark(
    dataset: CreditDataset,
    backend: Backend,
    config: ProjectedKernelConfig | None = None,
    test_size: float = 0.3,
    k_folds: int = 5,
    random_state: int = 42,
) -> dict[str, BenchmarkResult]:
    """Run the full credit scoring benchmark.

    Evaluates quantum kernel SVM, RBF-SVM, logistic regression,
    and (optionally) XGBoost on the given dataset.

    Parameters
    ----------
    dataset : CreditDataset
    backend : Quantum backend.
    config : Projected kernel config.
    test_size : Fraction of data for testing.
    k_folds : Number of CV folds.
    random_state : Random seed.

    Returns
    -------
    dict mapping classifier name to BenchmarkResult.
    """
    _require_sklearn()
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X, dataset.y, test_size=test_size,
        random_state=random_state, stratify=dataset.y,
    )

    results: dict[str, BenchmarkResult] = {}

    # Quantum kernel SVM
    qk_result = evaluate_quantum_kernel_svm(
        X_train, y_train, X_test, y_test, backend, config
    )
    results["quantum_kernel_svm"] = qk_result

    # Classical baselines
    for clf_name in ("rbf_svm", "logistic_regression"):
        result = evaluate_classical_baseline(
            X_train, y_train, X_test, y_test, classifier=clf_name
        )
        results[clf_name] = result

    # XGBoost if available
    if _HAS_XGBOOST:
        try:
            xgb_result = evaluate_classical_baseline(
                X_train, y_train, X_test, y_test, classifier="xgboost"
            )
            results["xgboost"] = xgb_result
        except Exception:
            warnings.warn("XGBoost evaluation failed, skipping.", stacklevel=2)

    return results
