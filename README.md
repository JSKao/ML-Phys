# ML-Phys: Machine Learning for Quantum Physics

A machine learning toolkit for quantum many-body systems and quantum information processing. This repository implements the **foundational quantum state generation and analysis** components that complement our [**QCVV (Quantum Characterization, Verification & Validation)**](https://github.com/JSKao/QCVV) framework for a complete quantum research pipeline.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Tests](https://img.shields.io/badge/Tests-38%20passed-brightgreen.svg)

## 🔗 Project Ecosystem

**ML-Phys** and **[QCVV](https://github.com/JSKao/QCVV)** work together to provide end-to-end quantum research capabilities:

- **ML-Phys** (this repo): Quantum state toolkit + ML architectures
- **[QCVV](https://github.com/JSKao/QCVV)**: Advanced verification protocols + experimental validation

As our research progresses toward **device-independent quantum certification**, these repositories will become increasingly integrated, with ML-Phys providing the theoretical foundations and QCVV handling the experimental verification.

## 🎯 Current Features

### Quantum State Generation
Complete toolkit for generating and analyzing quantum states:

```python
from quantum_graph_net.src.quantum.state_generator import QuantumStateGenerator
from quantum_graph_net.src.quantum.entanglement_metrics import EntanglementMetrics

# Generate various quantum states
gen = QuantumStateGenerator(3)
ghz_state = gen.ghz_state()
bell_state = gen.bell_state("phi_plus")
cluster_state = gen.cluster_state_1d()

# Comprehensive entanglement analysis
metrics = EntanglementMetrics(3)
entropy = metrics.entanglement_entropy(ghz_state, [0])
concurrence = metrics.concurrence(bell_state)
negativity = metrics.negativity(bell_state, [0])
```

**Supported States:**
- Bell states (all 4 types)
- GHZ states (arbitrary qubits)
- W states 
- Cluster states (1D chains)
- Arbitrary graph states
- Random pure states (Haar measure)

**Entanglement Measures:**
- Von Neumann entropy
- Entanglement entropy
- Concurrence (2-qubit)
- Negativity & logarithmic negativity
- Schmidt decomposition & rank
- Meyer-Wallach measure
- Multipartite entanglement metrics

### Data Loading & ML Pipeline

Universal data loader supporting multiple quantum data formats:

```python
from data_loader import load_data

# Load quantum state datasets
x_train, x_test, y_train, y_test, N = load_data("quantum_states.npz")

# Load graph structure data
graph = load_data("entanglement_graph.npz", graph_key="adjacency")

# Support for csv, npy, h5 formats
x_train, x_test, y_train, y_test, N = load_data("data.csv", N=100)
```

### Machine Learning Models

Modular PyTorch-based architectures for quantum ML:

```
models/
├── base_models.py      # Abstract base classes
├── cnn_model.py        # Convolutional architectures  
├── gnn_model.py        # Graph neural networks
├── layers.py           # Custom quantum-aware layers
└── trainer.py          # Training utilities
```

## 🧪 Testing & Verification

Comprehensive testing ensures mathematical correctness (38 tests, 100% pass rate):

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/ -m quantum    # Quantum algorithms
pytest tests/ -m unit       # Unit tests

# Interactive demonstration
python demo_quantum_enhancements.py
```

## 📊 Quick Demo

```bash
python demo_quantum_enhancements.py
```

This demonstrates:
- Quantum state generation across multiple types
- Entanglement analysis with various metrics
- State verification and fidelity calculations
- Integration with ML pipelines

## 🏗️ Repository Structure

```
ML-Phys/
├── quantum_graph_net/           # Main quantum ML package
│   ├── src/quantum/            
│   │   ├── state_generator.py   # Quantum state generation
│   │   └── entanglement_metrics.py  # Entanglement analysis
│   └── src/models/             # ML model architectures
├── models/                     # PyTorch implementations
├── tests/                      # Test suite (38 tests)
├── data/                       # Data loaders
├── docs/                       # Documentation
└── demo_quantum_enhancements.py  # Interactive demo
```

## 🔬 Research Applications

This toolkit enables research in:
- **Quantum Machine Learning**: Feature extraction from quantum states
- **Many-Body Physics**: Entanglement phase transitions
- **Quantum Information**: Resource quantification
- **Device Characterization**: State tomography preparation

## 🚀 Future Integration

As we develop toward **quantum resource certification**, this repository will increasingly integrate with **[QCVV](../QCVV)** to provide:

- **State Preparation** (ML-Phys) → **Experimental Verification** (QCVV)
- **Theoretical Analysis** (ML-Phys) → **Device Benchmarking** (QCVV)  
- **ML Predictions** (ML-Phys) → **Protocol Validation** (QCVV)

The combined ecosystem will support end-to-end quantum research from theoretical foundations to experimental validation.

## 📈 Installation & Requirements

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- `torch>=1.12.0` - PyTorch for ML models
- `numpy>=1.21.0` - Numerical computing
- `scipy>=1.7.0` - Scientific computing
- `pytest>=7.0.0` - Testing framework
