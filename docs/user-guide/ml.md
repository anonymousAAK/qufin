# Machine Learning

qufin includes quantum machine learning modules for financial applications including classification, generation, and feature extraction.

## Quantum Kernel Methods

Quantum kernels compute inner products in a quantum feature space that may capture complex correlations classical kernels miss.

```python
from qufin.ml.kernels import QuantumKernelClassifier, quantum_kernel_matrix
from qufin.backends.qiskit_backend import QiskitAerBackend

backend = QiskitAerBackend(method="automatic", seed=42)

# End-to-end SVM-style classifier backed by a quantum (ZZFeatureMap) kernel.
clf = QuantumKernelClassifier(n_qubits=4, backend=backend, reps=2)
clf.fit(X_train, y_train)        # X_train shape: (n_samples, n_qubits)
predictions = clf.predict(X_test)

# Or compute the kernel (Gram) matrix directly:
K_train = quantum_kernel_matrix(X_train, n_qubits=4, backend=backend, reps=2)
```

The kernel uses a `ZZFeatureMap` encoding with `reps` entangling layers.

## Variational Quantum Classifier (VQC)

A parameterized quantum circuit trained end-to-end for classification tasks.

```python
from qufin.ml.classifiers import VariationalQuantumClassifier, VQCConfig
from qufin.backends.qiskit_backend import QiskitAerBackend

config = VQCConfig(n_qubits=4, n_layers=3, optimizer="COBYLA")
clf = VariationalQuantumClassifier(config, QiskitAerBackend(method="automatic", seed=42))

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```

### Financial Applications

- **Credit scoring**: Classify loan applicants as default/non-default
- **Regime detection**: Classify market state as bull/bear/sideways
- **Fraud detection**: Identify anomalous transactions

## Quantum GAN (qGAN)

Quantum generative adversarial network for learning and sampling from probability distributions.

```python
from qufin.ml.qgan import QuantumGAN

qgan = QuantumGAN(
    n_qubits=4,
    generator_layers=3,
    discriminator_hidden=[64, 32],
    learning_rate=0.001,
)

# Train on historical return distribution
qgan.fit(returns_data, epochs=1000, batch_size=64)

# Generate synthetic samples
synthetic_returns = qgan.sample(n_samples=10000)
```

### Use Cases

- **Synthetic data generation**: Create realistic return distributions for backtesting
- **Privacy-preserving data sharing**: Share statistical properties without raw data
- **Data augmentation**: Expand small datasets for training other models

## Quantum Reservoir Computing

Uses a fixed quantum circuit as a dynamical reservoir, with only the readout layer trained classically. Computationally cheaper than VQC since the quantum parameters are not optimized.

```python
from qufin.ml.reservoir import QuantumReservoir

reservoir = QuantumReservoir(
    n_qubits=6,
    n_layers=4,
    readout="ridge",  # Ridge regression readout
)

# Time series prediction
reservoir.fit(X_train_seq, y_train)
predictions = reservoir.predict(X_test_seq)
```

## Model Comparison

| Model | Trainable Params | Training Cost | Best For |
|---|---|---|---|
| Quantum kernel | 0 (kernel only) | O(N^2) kernel matrix | Small datasets, high-dim features |
| VQC | O(qubits * layers) | Variational optimization | Classification with limited data |
| qGAN | Generator + discriminator | Adversarial training | Distribution learning |
| Reservoir | Readout only | Single regression | Time series, fast training |

## Tips

!!! tip "Start classical, go quantum"
    Always benchmark against a classical baseline (logistic regression, SVM, XGBoost) first. Quantum ML currently shows advantage primarily on small, highly correlated datasets.

!!! warning "NISQ limitations"
    Current quantum ML models are limited to 4-10 qubits on real hardware. Use simulators for larger circuits during development.
