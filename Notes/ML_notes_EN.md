# ✅ Core Structure of Machine Learning and Tensor Formats

## 📌 Core Three Elements and Extended Stages

The three fundamental elements of machine learning are:

**ML = Data + Model + Learning Mechanism**

To fully cover the entire machine learning lifecycle, you also need:

* **Evaluation & Validation**
* **Deployment & Maintenance**

## ✅ 1️⃣ Data

### 🎯 Definition

Raw information is transformed through **preprocessing** and **feature engineering** into a numerical representation that the model can compute — namely, a **tensor**.

### 📐 Common Preprocessing Formulas

**Standardization**

```
z = (x - μ) / σ
```

**Normalization**

```
x' = (x - x_min) / (x_max - x_min)
```

**One-Hot Encoding:** Convert categorical variables into binary vectors.

### 🧩 Feature Engineering & Dimensionality Reduction

* **Feature Engineering:** Create new, more meaningful features from raw data, or transform existing features.
  - **Dimensionality Reduction:** Techniques like PCA or SVD project high-dimensional data into lower dimensions while preserving key information.
    - **PCA:** Projects data onto directions of maximum variance.
    - **SVD:**

      ```
      M = U Σ V*
      ```

### 📦 Tensor Hierarchy

| Type   | Dimensionality | Example |
|--------|----------------|---------|
| Scalar | 0D             | `5` |
| Vector | 1D             | `x ∈ ℝ^F` |
| Matrix | 2D             | `M ∈ ℝ^{R × C}` |
| Tensor | 3D+            | `T ∈ ℝ^{D₁ × D₂ × … Dₖ}` |

### 📊 Input Tensor Formats for Different Models

| Model | Single Sample | Batch Format |
|------------------------|----------------|-----------------------------|
| **Traditional ML** | `x ∈ ℝ^F` | `X ∈ ℝ^{B × F}` |
| **DNN / MLP** | `x ∈ ℝ^F` | `X ∈ ℝ^{B × F}` |
| **CNN** | `X ∈ ℝ^{C × H × W}` | `X ∈ ℝ^{B × C × H × W}` |
| **GNN** | Node: `X ∈ ℝ^{N × F}`<br>Adjacency: `A ∈ {0,1}^{N × N}`<br>Edge list: `E ∈ ℤ^{2 × M}`<br>Edge features: `E_feat ∈ ℝ^{M × Fₑ}` | — |
| **RNN / LSTM** | `X ∈ ℝ^{L × F}` | `X ∈ ℝ^{B × L × F}` |
| **Transformer / LLM** | `X ∈ ℝ^{L × D_model}` | `X ∈ ℝ^{B × L × D_model}` |

### ⚛️ Quantum Machine Learning (QML)

| Type | Format | Description |
|------|--------|-------------|
| **Quantum-Enhanced ML** | `|ψ⟩ ∈ ℂ^{2ⁿ}` or `ρ ∈ ℂ^{2ⁿ × 2ⁿ}` | Quantum state vector or density matrix; classical data encoded by basis, angle, or amplitude. |
| **QRC** | `x(t) ∈ ℝ^F` | Only output layer trained; quantum reservoir does not require backpropagation. |
| **QAM** | — | Patterns stored in quantum state evolution without gradient descent. |

### 📚 GNN & QML Tensor Structures and Message Passing

#### 🧩 GNN Tensor Input & Message Passing

**Components:**

1. Node Features `(num_nodes, feature_dim)`
2. Adjacency Matrix `(num_nodes, num_nodes)` or Edge List `(2, num_edges)`
3. Edge Features `(num_edges, edge_feature_dim)`

**Message Passing:**

- Message: `m_ij^(l) = Message(h_i^(l), h_j^(l), e_ij)`
- Aggregate: `m_i^(l) = Aggregate({m_ij^(l) : j ∈ N(i)})`
- Update: `h_i^(l+1) = Update(h_i^(l), m_i^(l))`

> GNNs keep graph structure via multiple tensors and message passing.

#### 🧬 QML Tensor Supplement

- **Qubit:** `|ψ⟩ = α|0⟩ + β|1⟩`, `|α|² + |β|² = 1`
- **Density Matrix:** `ρ = |ψ⟩⟨ψ|` (pure) or `ρ = Σ pᵢ|ψᵢ⟩⟨ψᵢ|` (mixed)
- **Quantum Graph States:**

  ```
  |G⟩ = ∏ U_z(u,v) |+⟩^{⊗n}
  ```

- **Encoding:**
  - Basis Encoding
  - Angle Encoding
  - Amplitude Encoding

## ✅ 2️⃣ Model

**Definition:** A mathematical function mapping inputs to outputs with trainable parameters.

- Neuron: `a = σ(w ⋅ x + b)`
- Dense: `output = activation(XW + b)`
- Convolution: `output_ij = Σ input_{i+k,j+l} ⋅ kernel_{kl}`
- RNN:

  ```
  h_t = f(W_hh h_{t-1} + W_xh x_t + b)
  y_t = g(W_hy h_t + b)
  ```

- Transformer Attention:

  ```
  Q = XW^Q, K = XW^K, V = XW^V
  Attention(Q,K,V) = softmax(QKᵗ/√d_k) V
  ```

## ✅ 3️⃣ Learning Mechanism

**Definition:** Optimizes parameters to minimize loss.

### 🌱 Core Process

1. Forward: generate predictions.
2. Loss: `L = (1/N) Σ (yᵢ - ŷᵢ)²`
3. Backpropagation:

  ```
  ∂L/∂w_ij = ∂L/∂a_j ⋅ ∂a_j/∂z_j ⋅ ∂z_j/∂w_ij
  ```

4. Gradient Descent:

  ```
  w_{n+1} = w_n - α ∇L(w_n)
  ```

5. Adam Optimizer:

  ```
  m_t = β₁ m_{t-1} + (1-β₁)g_t
  v_t = β₂ v_{t-1} + (1-β₂)g_t²
  m̂_t = m_t / (1 - β₁ᵗ)
  v̂_t = v_t / (1 - β₂ᵗ)
  w_t = w_{t-1} - α m̂_t / (√v̂_t + ε)
  ```

## ✅ 4️⃣ Evaluation & Validation

- Accuracy, Precision, Recall, F1, ROC AUC
- Cross-Validation
- Interpretability

## ✅ 5️⃣ Deployment & Maintenance

- Deploy via APIs
- Monitor data drift, model degradation
- Retrain when needed
- Use GPU / TPU / FPGA acceleration

## 🔑 Summary

> **Everything starts with tensors, mapped by models, optimized through learning mechanisms, and wrapped up with evaluation and deployment.**

## 📖 Keywords

Tensor | Message Passing | Adjacency Matrix | Quantum Reservoir Computing (QRC) | Variational Quantum Algorithm (VQA) | Backpropagation | Gradient Descent | Adam Optimizer
