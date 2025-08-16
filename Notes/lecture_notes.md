# ✅ 機器學習核心構成與張量格式整理

---

## 📌 核心三要素與擴展環節

機器學習最核心的三大元素是：

**ML = 資料 (Data) + 模型 (Model) + 學習機制 (Learning Mechanism)**

若要完整覆蓋 ML 系統的生命週期，可再加上：

* **評估與驗證 (Evaluation & Validation)**
* **部署與維護 (Deployment & Maintenance)**

---

## ✅ 一、資料 (Data)

### 🎯 定義

原始資訊經過 **預處理**、**特徵工程**，最終轉換為模型可計算的 **數值表示**，也就是 **張量 (Tensor)**。

---

### 📐 常見預處理公式

**標準化 (Standardization)**


$z = \frac{x - \mu}{\sigma}$


**正規化 (Normalization)**


$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$

**獨熱編碼 (One-Hot Encoding)**：將類別變數轉為二進位向量。

---

### � 特徵工程與降維補充

* **特徵工程 (Feature Engineering)**：從原始資料中建立新的、更有意義的特徵，或對現有特徵進行轉換。
  - **降維 (Dimensionality Reduction)**：如主成分分析 (PCA) 或奇異值分解 (SVD)，將高維資料投影到低維空間，同時保留重要資訊。
    - **PCA**：將資料投影到最大變異方向。
    - **SVD** 數學表示：
      $$
      \mathbf{M} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^*
      $$
      其中 $\mathbf{M}$ 是原始資料矩陣，$\mathbf{U}$ 和 $\mathbf{V}^*$ 是么正矩陣，$\mathbf{\Sigma}$ 是奇異值對角矩陣。


### �📦 張量的階層

| 類型          | 維度  | 例子                                                                 |
| ----------- | --- | ------------------------------------------------------------------ |
| 標量 (Scalar) | 0D  | $5$                                                              |
| 向量 (Vector) | 1D  | $\mathbf{x} \in \mathbb{R}^F$                                    |
| 矩陣 (Matrix) | 2D  | $\mathbf{M} \in \mathbb{R}^{R \times C}$                         |
| 張量 (Tensor) | 3D+ | $\mathbf{T} \in \mathbb{R}^{D_1 \times D_2 \times \dots D_k}$ |


### 📊 不同模型的輸入張量格式

| 模型                         | 單樣本                                                                                                                                                                                                                                   | 批次                                                           |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| **傳統 ML (迴歸/SVM/K-Means)** | $\mathbf{x} \in \mathbb{R}^F$                                                                                                                                                                                                       | $\mathbf{X} \in \mathbb{R}^{B \times F}$                   |
| **DNN / MLP**              | $\mathbf{x} \in \mathbb{R}^F$                                                                                                                                                                                                       | $\mathbf{X} \in \mathbb{R}^{B \times F}$                   |
| **CNN**                    | $\mathbf{X} \in \mathbb{R}^{C \times H \times W}$                                                                                                                                                                                   | $\mathbf{X} \in \mathbb{R}^{B \times C \times H \times W}$ |
| **GNN**                    | Node: $\mathbf{X} \in \mathbb{R}^{N \times F}$<br>Adjacency: $\mathbf{A} \in {0,1}^{N \times N}$<br>Edge list: $\mathbf{E} \in \mathbb{Z}^{2 \times M}$<br>Edge features: $\mathbf{E}_{feat} \in \mathbb{R}^{M \times F_e}$ | —                                                            |
| **RNN / LSTM**             | $\mathbf{X} \in \mathbb{R}^{L \times F}$                                                                                                                                                                                            | $\mathbf{X} \in \mathbb{R}^{B \times L \times F}$          |
| **Transformer / LLM**      | Token: $\mathbf{X} \in \mathbb{R}^{L \times D_{model}}$                                                                                                                                                                            | $\mathbf{X} \in \mathbb{R}^{B \times L \times D_{model}}$ |

---

### ⚛️ 量子機器學習 (QML)

| 類型                                    | 格式                                 | 描述                                                                             |                            |
| ------------------------------------- | ---------------------------------- | ------------------------------------------------------------------------------ | -------------------------- |
| **量子增強型 ML**                          | $\ket{\psi} \in \mathbb{C}^{2^n}$ 或 $\rho \in \mathbb{C}^{2^n \times 2^n}$  | 量子態向量或密度矩陣，經典資料可用基底編碼、角度編碼、振幅編碼等方式嵌入 |  |
| **Quantum Reservoir Computing (QRC)** | $\mathbf{x}(t) \in \mathbb{R}^F$ | 只訓練輸出層，內部量子態不需反向傳播                                                             |                            |
| **Quantum Associative Memory (QAM)**  | —                                  | 模式存於量子態演化，不需梯度下降                                                               |                            |

### 📚 GNN & QML Tensor Structures and Message Passing (補充)

#### 🧩 GNN Tensor Input & Message Passing

* **Input Structure**: Graphs are non-Euclidean data, usually without fixed order or grid.
* **Input Components**:
  1. **Node Feature Matrix (X)**
     * Shape: `(num_nodes, feature_dimension)`
     * Each row is a node feature vector.
  2. **Adjacency Matrix (A)** or **Edge List**
     * Adjacency: `(num_nodes, num_nodes)`
     * Edge list (PyTorch Geometric): `(2, num_edges)`, source/target indices.
  3. **Edge Feature Matrix (E)** (if edges have attributes)
     * Shape: `(num_edges, edge_feature_dimension)`

* **Message Passing Mechanism**:
  - Each node builds messages from itself, neighbors, and edge features.
  - Nodes aggregate neighbors' messages (sum/mean/max).
  - Node updates its representation based on aggregated messages.

* **Message Passing Formulas:**
  - 訊息建立 (Message construction):
    $$
    m_{ij}^{(l)} = \text{Message}(h_i^{(l)}, h_j^{(l)}, e_{ij})
    $$
  - 訊息聚合 (Message aggregation):
    $$
    m_i^{(l)} = \text{Aggregate}({m_{ij}^{(l)} : j \in N(i)})
    $$
  - 節點更新 (Node update):
    $$
    h_i^{(l+1)} = \text{Update}(h_i^{(l)}, m_i^{(l)})
    $$

> GNNs do not flatten the whole graph into a single vector, but use multiple tensors and message passing to capture structure.

#### 🧬 QML Tensor Structure Supplement

* **Quantum-Enhanced ML**
  * Encoded state vector: $|\psi\rangle \in \mathbb{C}^{2^n}$
  * Density matrix: $\rho \in \mathbb{C}^{2^n \times 2^n}$
  * Encoding: basis, angle, amplitude

* **Quantum Reservoir Computing (QRC)**
  * Input: $\mathbf{x}(t) \in \mathbb{R}^F$
  * Only output layer is trained; internal quantum state evolution does not require backpropagation.

* **Quantum Associative Memory (QAM)**
  * Patterns are stored in quantum state evolution, no gradient descent required.

#### 量子資料的理論表示與編碼補充

* **量子位元 (Qubit)**：量子資訊的基本單位。單一 qubit 純態：
  $|\psi\rangle = \alpha |0\rangle + \beta |1\rangle,\quad \alpha, \beta \in \mathbb{C},\ |\alpha|^2 + |\beta|^2 = 1$
  多 qubit 系統狀態以張量積組合。

* **密度矩陣 (Density Matrix)**：描述純態或混合態。n qubit 系統密度矩陣 $\rho$ 為 $2^n \times 2^n$ Hermitian 矩陣，$\text{Tr}(\rho) = 1$。
  - 純態：$\rho = |\psi\rangle\langle\psi|$
  - 混合態：$\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|$

* **量子圖態 (Quantum Graph States)**：將數學圖表示為量子態。
  - 無權重圖 $G=(V,E)$：
    
    $|G\rangle = \prod_{(u,v)\in E} U_z(u,v) |+\rangle^{\otimes n}$
    
    其中 $U_z(u,v)$ 為受控 Z 門。
  - 加權圖：
    $U_z(u,v,w) = e^{-iw\sigma_z^u \sigma_z^v}$
    
    $\sigma_z$ 為 Pauli Z 矩陣，w 為邊權重。

* **經典資料到量子態的編碼 (Classical-to-Quantum Data Encoding)**：
  - **基底編碼 (Basis Encoding)**：將經典二進位資料直接映射到 qubit 計算基底態。
    
    $|b_1 b_2 \ldots b_F\rangle = |b_1\rangle \otimes |b_2\rangle \otimes \ldots \otimes |b_F\rangle$
    
  - **角度編碼 (Angle Encoding)**：將資料值編碼為量子門旋轉角度。
  - **振幅編碼 (Amplitude Encoding)**：將資料向量正規化後作為量子態的振幅。

## ✅ 二、模型 (Model)

**定義**：以數學函數形式連接輸入到輸出，包含可訓練參數。

* **神經元 (Neuron)**

  
  $a = \sigma(\mathbf{w} \cdot \mathbf{x} + b)$
  

* **密集層 (Dense)**

  
  $\text{output} = \text{activation}(\mathbf{XW} + b)$
  

* **卷積層 (CNN)**

  
  $\text{output}_{i,j} = \sum_{k,l} \text{input}_{i+k, j+l} \cdot \text{kernel}_{k,l}$
  

* **循環層 (RNN)**

  
  $\mathbf{h}_t = f(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + b)$
  

  
  $\mathbf{y}_t = g(\mathbf{W}_{hy} \mathbf{h}_t + b)$
  

* **Transformer Attention**

  
  $\mathbf{Q} = \mathbf{X} \mathbf{W}^Q,\quad 
  \mathbf{K} = \mathbf{X} \mathbf{W}^K,\quad 
  \mathbf{V} = \mathbf{X} \mathbf{W}^V$
  

  
  $\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V}$
  

  - 其中 $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ 分別為 Query、Key、Value 矩陣，$\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V$ 為權重矩陣，$d_k$ 為 Key 向量維度。

---

## ✅ 三、學習機制 (Learning Mechanism)

**定義**：優化可學參數以最小化損失。

---

### 🌱 核心流程

1️⃣ **前向傳播 (Forward Propagation)**
輸入資料經網路產生預測。

2️⃣ **損失函數 (Loss Function)**
衡量預測與真值的差異，例如均方誤差：


$L = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$


3️⃣ **反向傳播 (Backpropagation)**
用鏈式法則計算梯度：


$\frac{\partial L}{\partial w_{ij}} =
\frac{\partial L}{\partial a_j} \cdot
\frac{\partial a_j}{\partial z_j} \cdot
\frac{\partial z_j}{\partial w_{ij}}$


4️⃣ **梯度下降 (Gradient Descent)**
根據梯度調整參數：


$\mathbf{w}_{n+1} = \mathbf{w}_n - \alpha \nabla L(\mathbf{w}_n)$


5️⃣ **優化器 (如 Adam)**
在基本梯度下降上引入動量、自適應學習率與偏差修正：


$\begin{aligned}
\mathbf{m}_t &= \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t \\
\mathbf{v}_t &= \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2 \\
\hat{\mathbf{m}}_t &= \frac{\mathbf{m}_t}{1 - \beta_1^t} \\
\hat{\mathbf{v}}_t &= \frac{\mathbf{v}_t}{1 - \beta_2^t} \\
\mathbf{w}_t &= \mathbf{w}_{t-1} - \alpha \cdot \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}
\end{aligned}$


---

✅ **重點**

* 反向傳播：計算梯度的方法
* 梯度下降：用梯度更新參數的策略
* Adam：一種進階的梯度下降變體
  → 共同構成訓練迴圈，直到收斂。

---

## ✅ 四、評估與驗證 (Evaluation & Validation)

* Accuracy, Precision, Recall, F1, ROC AUC
* Cross-Validation
* 可解釋性 (Interpretability)

---

## ✅ 五、部署與維護 (Deployment & Maintenance)

* 以 API 部署模型
* 監控資料漂移與模型退化
* 需要時進行再訓練
* GPU / TPU / FPGA 加速

---

## 🔑 總結

> **一切從張量開始，依靠模型進行映射，透過學習機制優化，最後以嚴謹的評估驗證與有效部署收尾。**

---

## 📖 Keywords

Tensor ｜ Message Passing ｜ Adjacency Matrix ｜ Quantum Reservoir Computing (QRC) ｜ Variational Quantum Algorithm (VQA) ｜ Backpropagation ｜ Gradient Descent ｜ Adam Optimizer

---

