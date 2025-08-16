## `QuantumGraphNet - GNN Tasks`

**內容摘要**：

- 糾纏分類（Entanglement Classifier）
    
- 糾纏度量預測（回歸任務）
    
- 操作/測量後狀態預測
    
- 為每個任務對應適合的模型與訓練流程


## GNN in Quantum Information

- 範式使用：
  - ✅ 監督式：分類量子態、預測糾纏度量
  - ✅ 自監督：表徵學習、對比學習
  - 🔄 非監督：少量應用於拓撲分類
  - 🔄 強化式：探索中，用於電路設計與優化

- 任務範例：
  - Entanglement classification
  - Concurrence prediction
  - Topological phase classification
  - Quantum error correction
  - Circuit compression

📌 主要用途歸納如下：

| ML 範式      | GNN 在量子資訊的典型任務                                     | 例子/應用                                                       |
| ---------- | -------------------------------------------------- | ----------------------------------------------------------- |
| **監督式學習**  | 🔹 分類量子態（如 GHZ vs W）🔹 預測量子糾纏度量🔹 預測哈密頓量特性         | - Entanglement Classification- Concurrence / Negativity 預測  |
| **自監督學習**  | 🔹 表徵學習（pre-training quantum graphs）🔹 模仿量子操作的對比學習 | - GraphCL for Quantum Circuits- GNN-based MAE               |
| **非監督式學習** | 🔹 探索量子系統聚類、拓撲相分類                                  | - Cluster different phase regimes (e.g., symmetry-breaking) |
| **強化學習**   | 🔹 探索量子電路設計與壓縮策略                                   | - RL-guided quantum circuit synthesis                       |
🧠 任務類型詳細總表：

| 類別          | GNN任務                                            | 說明          |
| ----------- | ------------------------------------------------ | ----------- |
| **量子態分類**   | GNN 輸入圖表示的量子態 → 輸出態類型（GHZ、W等）                    | 監督式         |
| **糾纏度量預測**  | 預測 concurrence、negativity、entanglement entropy 等 | 監督式         |
| **量子態生成模型** | 預測/模擬量子狀態的機率分布                                   | 自監督、生成模型    |
| **量子相圖學習**  | 分辨不同拓撲相（topological phase）                       | 監督 / 非監督    |
| **量子電路學習**  | 將量子電路編碼成圖，進行優化與壓縮                                | 自監督 / 強化式   |
| **量子錯誤糾正**  | 對錯誤路徑進行圖表示與預測                                    | 自監督 or 強化學習 |
| **量子圖卷積**   | 對 Qubit 間交互作用網路進行訊息傳遞                            | 作為模組使用      |

### 🔧 舉個實例：

例如你有一個 `n`-qubit 的 GHZ 态：

$∣GHZn⟩=12(∣00...0⟩+∣11...1⟩)∣GHZn​⟩=2​1​(∣00...0⟩+∣11...1⟩)$

你可以把 qubit 間的關聯表示為圖，然後用 GNN 來：

- **分類這是否為 GHZ / Cluster / W 态**（監督）
    
- **預測它的糾纏熵**（監督）
    
- **建立語意嵌入表徵**（自監督）
    
- **設計更好的電路來實現它**（強化）



