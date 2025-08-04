# Variational Autoencoder (VAE)
## 原理、數學推導與演算法細節流程

1. VAE 原理概述
VAE 是一種生成模型，能學習資料的潛在（latent）結構，並能生成新資料。
它結合了自編碼器（Autoencoder）與變分推斷（Variational Inference）。
目標：學習 $p(x)$，即資料 $x$ 的生成分布。

2. 模型架構
Encoder（推斷網路）：$q_\phi(z|x)$，將資料 $x$ 映射到潛在變數 $z$ 的分布（通常是高斯分布，$\mu(x), \sigma(x)$ 由神經網路給出）。
Decoder（生成網路）：$p_\theta(x|z)$，從 $z$ 生成資料 $x$（通常也是神經網路）。

3. 數學推導
(1) Evidence Lower Bound (ELBO)
我們想最大化 $\log p_\theta(x)$，但這個積分通常不可解：

$$ \log p_\theta(x) = \log \int p_\theta(x|z) p(z) dz $$

引入近似後驗 $q_\phi(z|x)$，有：

$$ \log p_\theta(x) = \mathbb{E}{q\phi(z|x)} \left[ \log \frac{p_\theta(x|z)p(z)}{q_\phi(z|x)} \right] + D_{KL}(q_\phi(z|x) | p_\theta(z|x)) $$

其中 $D_{KL}$ 是 KL divergence，非負。因此：

$$ \log p_\theta(x) \geq \mathbb{E}{q\phi(z|x)} [\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) | p(z)) $$

這個下界稱為 ELBO，我們最大化它。

(2) VAE 的 Loss Function
對單一資料 $x$，VAE 的 loss 為：

$$ \mathcal{L}{VAE} = -\mathbb{E}{q_\phi(z|x)} [\log p_\theta(x|z)] + D_{KL}(q_\phi(z|x) | p(z)) $$

第一項：重建誤差（reconstruction loss），衡量 decoder 生成 $x$ 的能力。
第二項：KL loss，讓 encoder 輸出的 $q_\phi(z|x)$ 不要偏離 prior $p(z)$（通常是標準高斯）。

(3) 重參數化技巧（Reparameterization Trick）
$q_\phi(z|x) = N(z; \mu(x), \sigma^2(x))$
直接 sampling $z$ 不可微分，無法反向傳播。
解法：$z = \mu(x) + \sigma(x) \cdot \epsilon$，$\epsilon \sim N(0,1)$
這樣 $z$ 對 $\mu, \sigma$ 可微分，loss 可以傳回 encoder。

4. VAE 訓練流程（Pseudocode）
輸入資料 $x$
Encoder 輸出 $\mu(x), \log \sigma^2(x)$
Sampling $\epsilon \sim N(0,1)$，計算 $z = \mu + \sigma \cdot \epsilon$
Decoder 用 $z$ 生成 $\hat{x} = p_\theta(x|z)$
計算 loss：
重建誤差：$-\log p_\theta(x|z)$（通常用 MSE 或 BCE）
KL loss：$D_{KL}(N(\mu, \sigma^2) | N(0,1)) = \frac{1}{2} \sum (1 + \log \sigma^2 - \mu^2 - \sigma^2)$
總 loss = 重建誤差 + KL loss
反向傳播，更新 encoder/decoder 參數


5. VAE Loss 的數學細節
KL divergence between $N(\mu, \sigma^2)$ and $N(0,1)$：

$$ D_{KL}(N(\mu, \sigma^2) | N(0,1)) = \frac{1}{2} \sum_{i=1}^d \left( \mu_i^2 + \sigma_i^2 - \log \sigma_i^2 - 1 \right) $$


x ──► Encoder ──► μ, logσ² ──► z = μ + σ·ε ──► Decoder ──► x̂
         ▲                                 │
         └───────────── loss ◄─────────────┘


7. VAE 生成新資料
訓練好後，直接從 $N(0,1)$ sampling $z$，丟給 decoder 產生新 $x$。


8. 簡單 PyTorch 風格程式片段

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, ...):
        ...
    def forward(self, x):
        mu = ...
        logvar = ...
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, ...):
        ...
    def forward(self, z):
        x_hat = ...
        return x_hat

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps

# 訓練步驟
mu, logvar = encoder(x)
z = reparameterize(mu, logvar)
x_hat = decoder(z)
recon_loss = F.mse_loss(x_hat, x)
kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
loss = recon_loss + kl_loss
loss.backward()


# Graph Neural Network(GNN)

圖神經網絡（Graph Neural Network, GNN）是一種能夠處理圖結構數據的機器學習模型。與傳統的神經網絡（如 CNN 處理影像、RNN 處理序列）相比，GNN 的設計目的是解決節點之間非結構化連接（如社交網絡、知識圖譜、分子結構）所帶來的學習挑戰。

---

## 🔍 一、GNN 的基本原理

### 📌 核心任務：信息傳遞（Message Passing）

GNN 的核心思想是：每個節點不只看自己，還會考慮鄰居節點的資訊，來更新自己的表示（embedding）。

### 🧱 一般 GNN 的更新流程：

設圖為 \( G=(V,E) \)，其中：

- \( V \)：節點集合（nodes）
- \( E \)：邊集合（edges）
- \( h_v^{(k)} \)：節點 \( v \) 在第 \( k \) 層的特徵表示（embedding）

每一層 GNN 的步驟包含：

1. **Message aggregation（資訊彙整）**：

\[
m_v^{(k)} = \text{AGGREGATE}^{(k)}(\{ h_u^{(k-1)} : u \in N(v) \})
\]

其中 \( N(v) \) 是節點 \( v \) 的鄰居集合。  
Aggregate 可以是 mean, sum, max, attention 等。

2. **Update（狀態更新）**：

\[
h_v^{(k)} = \text{UPDATE}^{(k)}(h_v^{(k-1)}, m_v^{(k)})
\]

重複這個流程 \( K \) 次後，就能學出每個節點的最終表示 \( h_v^{(K)} \)，可以用來做：

- 節點分類
- 圖分類（如整個分子是否有毒）
- 邊預測（如知識圖譜補全）

---

## 🧠 二、與傳統機器學習模型對比

| 特性       | CNN（卷積神經網絡）         | RNN（循環神經網絡）      | GNN（圖神經網絡）            |
|------------|-----------------------------|--------------------------|-----------------------------|
| 資料結構   | 網格（如影像）              | 序列（如語音、文字）      | 圖結構（任意連接關係）       |
| 鄰近性結構 | 固定鄰域（如 3x3 kernel）   | 線性鄰居（前一個/下一個） | 任意鄰居（非線性）           |
| 參數共用   | 是                          | 是                       | 是（消息傳遞共用規則）       |
| 資訊傳遞方式 | 卷積                       | 時序傳遞                  | 聚合鄰居特徵後更新           |
| 與 GNN 最接近的是？ | 🔁 CNN（概念上最類似） |                          |                             |

🔸 GNN 和 CNN 最像：都強調「鄰近資料的聚合」，只是 CNN 是固定結構，而 GNN 可處理任意圖。

---

## 🧪 三、常見 GNN 模型類型

| 模型      | 聚合方式                 | 特點                       |
|-----------|--------------------------|----------------------------|
| GCN       | 鄰居平均 + 線性轉換       | 最經典、簡單               |
| GraphSAGE | Sum/Mean/Max + MLP        | 可處理大圖、支援 inductive |
| GAT       | 加入 self-attention 機制  | 自適應加權鄰居，提升效果   |
| GIN       | 使用 injective function   | 理論證明有更強表示能力     |

---

## 🔬 四、GNN 的應用場景

- 社交網絡：朋友推薦、社群偵測
- 分子圖分析：藥物設計、毒性預測
- 知識圖譜：推理、問答系統
- 程式分析：靜態分析、程式相似度
- 量子物理：分子表示學習、Hamiltonian 網絡建模

---

## 🛠 五、簡單 PyTorch Geometric 範例

```python
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class SimpleGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x
