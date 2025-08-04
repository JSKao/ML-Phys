# ✅ Machine Learning

## The three fundamental elements of machine learning:

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

- Message: $m_{ij}^{l}$ = Message$(h_{i}^{l}, h_{j}^{l}, e_{ij})$
- Aggregate: $m_{i}^{l}$ = Aggregate$({m_{ij}^{l} : j ∈ N(i)})$
- Update: $h_i^{l+1}$ = Update$(h_{i}^{l}, m_{i}^{l})$

> GNNs keep graph structure via multiple tensors and message passing.

#### 🧬 QML Tensor Supplement

- **Qubit:** `|ψ⟩ = α|0⟩ + β|1⟩`, `|α|² + |β|² = 1`
- **Density Matrix:** `ρ = |ψ⟩⟨ψ|` (pure) or `ρ = Σ pᵢ|ψᵢ⟩⟨ψᵢ|` (mixed)
- **Quantum Graph States:**

  
  $\ket{G} = \Pi U_{z}^{\{u,v\}} \ket{+}^{⊗n}$
  

- **Encoding:**
  - Basis Encoding
  - Angle Encoding
  - Amplitude Encoding

## ✅ 2️⃣ Model

**Definition:** A mathematical function mapping inputs to outputs with trainable parameters.

- Neuron: $a = \sigma(\mathbf{w} \cdot \mathbf{x} + b)$
- Dense: $\text{output} = \text{activation}(\mathbf{XW} + b)$
- Convolution: $\text{output}_{i,j} = \sum_{k,l} \text{input}_{i+k, j+l} \cdot \text{kernel}_{k,l}$
- RNN:

  $\mathbf{h}_t = f(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + b)$
  

  
  $\mathbf{y}_t = g(\mathbf{W}_{hy} \mathbf{h}_t + b)$

- Transformer Attention:

  $\mathbf{Q} = \mathbf{X} \mathbf{W}^Q,\quad 
    \mathbf{K} = \mathbf{X} \mathbf{W}^K,\quad 
    \mathbf{V} = \mathbf{X} \mathbf{W}^V$
  

  
  $\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V}$

   - where $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ are Query、Key、Value matrices respectively, $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V$ are weight matrices, $d_k$ is dimension of Key vector.

## ✅ 3️⃣ Learning Mechanism

**Definition:** Optimizes parameters to minimize loss.

### 🌱 Core Process

1. Forward: generate predictions.
2. Loss: $L = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$
3. Backpropagation:

  $\frac{\partial L}{\partial w_{ij}} =
   \frac{\partial L}{\partial a_j} \cdot
   \frac{\partial a_j}{\partial z_j} \cdot
   \frac{\partial z_j}{\partial w_{ij}}$

4. Gradient Descent:

  $\mathbf{w}_{n+1} = \mathbf{w}_n - \alpha \nabla L(\mathbf{w}_n)$


5. Adam Optimizer:

  $\begin{aligned}
   \mathbf{m}_t &= \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t \\
   \mathbf{v}_t &= \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2 \\
   \hat{\mathbf{m}}_t &= \frac{\mathbf{m}_t}{1 - \beta_1^t} \\
   \hat{\mathbf{v}}_t &= \frac{\mathbf{v}_t}{1 - \beta_2^t} \\
   \mathbf{w}_t &= \mathbf{w}_{t-1} - \alpha \cdot \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}
   \end{aligned}$

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