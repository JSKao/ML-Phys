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