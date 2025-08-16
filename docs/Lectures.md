

# **機器學習核心概念與量子應用講義**

## **I. 機器學習的基礎構成**

機器學習（ML）系統可以被歸納為三個核心元素：**資料 (Data)**、**模型 (Model)** 和 **學習機制 (Learning Mechanism)**。然而，一個完整的 ML 專案生命週期還需包含 **評估與驗證 (Evaluation & Validation)** 以及 **部署與維護 (Deployment & Maintenance)**。

### **1.1 資料 (Data)**

資料是機器學習的基石。無論其原始形式為何（如文字、圖像、音訊、圖結構等），在輸入模型進行計算時，都必須轉換為**數值張量 (Tensor)** 的形式。張量是多維陣列的泛稱，是所有現代深度學習框架處理資料的基礎。

* **定義**：原始資訊經過預處理、特徵工程後，轉換為模型可計算的數值表示，也就是張量。  
* **預處理 (Preprocessing)**：涉及資料清洗（處理缺失值、異常值）、正規化、標準化、編碼等。  
  * 標準化 (Standardization)：將特徵縮放到平均值為 0、標準差為 1。  
    z=σx−μ​  
    其中 x 是原始特徵，μ 是平均值，σ 是標準差，z 是標準化後的特徵。  
  * 正規化 (Normalization)：將特徵縮放到特定範圍（例如 0 到 1）。  
    x′=xmax​−xmin​x−xmin​​  
  * 獨熱編碼 (One-Hot Encoding)：將類別變數轉換為二進位向量。  
    例如，類別 "Red", "Blue", "Green" 可編碼為 , , \`\`。  
* **特徵工程 (Feature Engineering)**：從原始資料中建立新的、更有意義的特徵，或對現有特徵進行轉換。  
  * 降維 (Dimensionality Reduction)：如主成分分析 (PCA) 或奇異值分解 (SVD)，將高維資料投影到低維空間，同時保留重要資訊。  
    SVD 的數學表示：$ \\mathbf{M} \= \\mathbf{U} \\mathbf{\\Sigma} \\mathbf{V}^\* $  
    其中 M 是原始資料矩陣，U 和 V∗ 是么正矩陣，Σ 是奇異值對角矩陣。  
  * **資料增強 (Data Augmentation)**：透過對現有資料進行轉換（如圖像的旋轉、翻轉、裁剪；文字的同義詞替換、回譯等）來生成新的訓練樣本，以擴充資料集並提高模型的泛化能力。這對於資料稀缺或需要提高模型魯棒性的場景尤為重要。  
  * **自動化特徵工程 (Automated Feature Engineering, AutoFE)**：利用演算法自動探索和建立新的特徵，減少人工干預。這包括基於樹的方法、深度學習方法和進化演算法等。  
* **張量形式 (Tensor Form)**：  
  * **標量 (Scalar)**：單一數值，0D 張量。例如：5  
  * **向量 (Vector)**：有序數值列表，1D 張量。例如：$ \\mathbf{x} \= \[x\_1, x\_2, \\ldots, x\_F\] \\in \\mathbb{R}^F $  
  * **矩陣 (Matrix)**：2D 數值陣列。例如：$ \\mathbf{M} \\in \\mathbb{R}^{R \\times C} $ (R 行 C 列)  
  * **張量 (Tensor)**：多維數值陣列。例如：$ \\mathbf{T} \\in \\mathbb{R}^{D\_1 \\times D\_2 \\times \\ldots \\times D\_k} $  
* **不同模型對接的資料張量形式**：  
  * **傳統 ML (Traditional ML)** (如線性迴歸、支持向量機 SVM)：  
    * 單一樣本輸入：$ \\mathbf{x} \\in \\mathbb{R}^{F} $ (一個 F 維的向量)  
    * 批次樣本輸入：$ \\mathbf{X} \\in \\mathbb{R}^{B \\times F} $ (一個 B×F 的矩陣)，其中 B 是批次大小，F 是特徵數量。  
  * **深度神經網路 (Deep Neural Networks, DNN) / 多層感知器 (Multi-Layer Perceptrons, MLP)**：  
    * 單一樣本輸入：$ \\mathbf{x} \\in \\mathbb{R}^{F} $ (扁平化向量)  
    * 批次樣本輸入：$ \\mathbf{X} \\in \\mathbb{R}^{B \\times F} $ (矩陣)  
    * 「密集 (Dense)」的體現：DNN 的「密集層」之所以得名，是因為該層的每個神經元都與前一層的所有神經元相連接。這種全連接導致了大量的參數和密集的矩陣運算。  
      $ \\text{output} \= \\text{activation} (\\text{dot}(\\text{input}, \\text{kernel}) \+ \\text{bias}) $  
      其中 kernel 是權重矩陣。  
  * **卷積神經網路 (Convolutional Neural Networks, CNN)**：  
    * 單一圖像輸入：$ \\mathbf{X} \\in \\mathbb{R}^{C \\times H \\times W} $ (通道數 x 高度 x 寬度)  
    * 批次圖像輸入：$ \\mathbf{X} \\in \\mathbb{R}^{B \\times C \\times H \\times W} $ (批次 x 通道數 x 高度 x 寬度)  
    * **運作方式**：CNN 的卷積核（濾波器）在輸入張量上滑動，提取局部特徵。這些操作本質上是密集的矩陣運算，但由於卷積核的局部連接特性，參數數量比全連接層少得多。  
  * **圖神經網路 (Graph Neural Networks, GNN)**：  
    * **資料形式**：處理非歐幾里得的圖結構資料。圖由節點和邊組成。  
    * **數學表示**：GNN 的輸入通常是**多個張量的集合**，共同描述圖：  
      * **節點特徵矩陣 (Node Feature Matrix)**：$ \\mathbf{X} \\in \\mathbb{R}^{N \\times F} $，其中 N 是圖中節點的數量，F 是每個節點的特徵維度.  
      * **鄰接矩陣 (Adjacency Matrix)**：$ \\mathbf{A} \\in {0, 1}^{N \\times N} ，表示節點之間的連接關係。對於稀疏圖，通常以稀疏格式儲存，例如∗∗邊列表(EdgeList)∗∗： \\mathbf{E}\_{idx} \\in \\mathbb{Z}^{2 \\times M} $，其中 M 是邊的數量.  
      * **邊特徵矩陣 (Edge Feature Matrix)** (可選)：$ \\mathbf{E}\_{feat} \\in \\mathbb{R}^{M \\times F\_e} $，其中 Fe​ 是邊特徵維度.  
    * **運作方式**：GNN 透過「訊息傳遞 (Message Passing)」機制在圖上運作. 每個節點從其鄰居收集訊息（利用鄰接矩陣），聚合這些訊息，然後更新自己的特徵表示. 這些訊息轉換和節點更新步驟內部會涉及**密集矩陣運算**.  
      * **訊息建立**：$ m\_{ij}^{(l)} \= \\text{Message}(h\_i^{(l)}, h\_j^{(l)}, e\_{ij}) $ 7  
      * **訊息聚合**：$ m\_i^{(l)} \= \\text{Aggregate}({m\_{ij}^{(l)} : j \\in N(i)}) $ 7  
      * **節點更新**：$ h\_i^{(l+1)} \= \\text{Update}(h\_i^{(l)}, m\_i^{(l)}) $ 7  
  * **循環神經網路 (Recurrent Neural Networks, RNN) / 長短期記憶網路 (Long Short-Term Memory, LSTM)**：  
    * **資料形式**：處理序列資料，如文字、時間序列、音訊。  
    * **數學表示**：  
      * 單一序列輸入：$ \\mathbf{X} \\in \\mathbb{R}^{L \\times F} $ (序列長度 x 特徵維度)  
      * 批次序列輸入：$ \\mathbf{X} \\in \\mathbb{R}^{B \\times L \\times F} $ (批次 x 序列長度 x 特徵維度)  
    * **運作方式**：RNN 和 LSTM 透過循環連接處理序列，將前一個時間步的隱藏狀態作為當前時間步的輸入，從而捕捉時間依賴性. 它們的內部計算（如門控機制）也涉及密集矩陣運算。  
  * **Transformer / 大型語言模型 (Large Language Models, LLM)**：  
    * **資料形式**：主要處理序列資料，特別是文字，但也可以是圖像、音訊等  
    * **數學表示**：  
      * 輸入文字經過**分詞 (Tokenization)** 轉換為數字 ID 序列，再通過**嵌入層 (Embedding Layer)** 轉換為高維向量序列。  
      * 單一序列輸入：$ \\mathbf{X} \\in \\mathbb{R}^{L \\times D\_{model}} $ (序列長度 x 模型維度)。  
      * 批次序列輸入：$ \\mathbf{X} \\in \\mathbb{R}^{B \\times L \\times D\_{model}} $ (批次 x 序列長度 x 模型維度)。  
      * **位置編碼 (Positional Encoding)**：一個與輸入嵌入維度相同的向量，被加到每個 token 的嵌入向量中，以提供序列中的位置資訊。  
      * **自注意力機制 (Self-Attention)**：序列中的**每個 token 都會與序列中的所有其他 token 進行互動**，以計算其上下文相關的表示。  
        * Query ($ \\mathbf{Q} ),Key( \\mathbf{K} ),Value( \\mathbf{V} $) 矩陣由輸入嵌入與各自的權重矩陣相乘得到。  
        * 注意力分數：$ \\text{Attention}(\\mathbf{Q}, \\mathbf{K}, \\mathbf{V}) \= \\text{softmax}\\left(\\frac{\\mathbf{Q}\\mathbf{K}^{\\mathrm{T}}}{\\sqrt{d\_k}}\\right)\\mathbf{V} $。其中 dk​ 是 Key 向量的維度。  
      * **「密集 (Dense)」的體現**：LLM 的基礎 Transformer 架構，特別是自注意力機制和前饋網路，都是透過**密集的矩陣運算**來實現的。

### **1.1.1 資料品質與數量 (Data Quality and Quantity)**

資料的品質和數量對於機器學習模型的性能和泛化能力至關重要。

* **資料品質 (Data Quality)**：  
  * **定義**：指資料的準確性、完整性、一致性、時效性和相關性。  
  * **重要性**：低品質的資料會導致模型學習到錯誤的模式，產生有偏差或不準確的預測，甚至導致模型失效。例如，嘈雜的資料會降低模型性能。  
  * **挑戰**：真實世界的資料往往存在缺失值、異常值、不一致性、重複項和偏差。  
  * **處理方法**：資料清洗、異常值處理、缺失值填補、去重、偏差檢測與緩解等。  
* **資料數量 (Data Quantity)**：  
  * **定義**：指用於訓練模型的資料量。  
  * **重要性**：複雜的模型（特別是深度學習模型）通常需要大量的資料才能有效學習和泛化 9。資料量不足會導致模型欠擬合或泛化能力差 13。  
  * **挑戰**：某些領域（如量子實驗、醫療影像）的高品質標註資料可能稀缺且昂貴 15。  
  * **處理方法**：  
    * **資料增強 (Data Augmentation)**：如前所述，透過轉換現有資料來生成新樣本。  
    * **遷移學習 (Transfer Learning)**：利用在大型資料集上預訓練的模型，然後在較小的目標資料集上進行微調。  
    * **主動學習 (Active Learning)**：模型主動選擇最有資訊量的未標註資料點進行標註，以最小化標註成本。  
    * **合成資料生成 (Synthetic Data Generation)**：使用生成模型（如 GANs、擴散模型）來創建新的、逼真的資料樣本 19。

### **1.2 模型 (Model)**

模型是學習資料模式的數學函數或計算結構。

* **定義**：一個包含可學習參數的數學函數，用於從輸入資料映射到輸出預測。  
* 神經元 (Neuron)：基本計算單元，執行加權求和、加偏差和激活函數。  
  a=σ(w⋅x+b)  
  其中 x 是輸入向量，w 是權重向量，b 是偏差，σ 是激活函數，a 是輸出。  
* **激活函數 (Activation Function)**：引入非線性，使模型能夠學習複雜的模式。常見的激活函數包括 ReLU、Sigmoid 和 Tanh。  
* **層 (Layer)**：神經元的集合，對輸入進行轉換。  
  * 密集層 (Dense Layer)：每個輸入神經元連接到下一層的所有神經元。  
    $ \\text{output} \= \\text{activation} (\\text{dot}(\\text{input}, \\text{kernel}) \+ \\text{bias}) $  
  * **卷積層 (Convolutional Layer)**：使用濾波器（核）在局部區域內提取特徵。  
  * **循環層 (Recurrent Layer)**：具有內部隱藏狀態，處理序列資料並捕捉時間依賴性。  
  * **注意力層 (Attention Layer)**：動態地為輸入序列的不同部分分配權重，以捕捉上下文關係 26。  
* **模型架構 (Model Architecture)**：層次的組合方式（例如，編碼器-解碼器結構、多頭注意力機制）。

#### **III. 學習機制 (Learning Mechanism)**

這是模型如何從資料中學習並調整參數的過程。

* **定義**：用於調整模型參數以最小化損失函數的演算法。  
* **學習範式**：  
  * **監督式學習 (Supervised Learning)**：使用帶有標籤的資料集訓練模型，以分類或預測結果  
    * 目標：學習函數 $ y \= f(\\mathbf{x}) $，其中 $y$ 是輸出，$\\mathbf{x}$ 是輸入。  
  * **非監督式學習 (Unsupervised Learning)**：使用無標籤資料集，發現隱藏模式或資料分組。  
    * 目標：學習資料的機率分佈 $ p(\\mathbf{X}) $。  
  * **半監督式學習 (Semi-supervised Learning)**：結合少量標籤資料和大量無標籤資料 1。  
  * **強化學習 (Reinforcement Learning)**：代理通過與環境互動，接收獎勵或懲罰來學習決策策略。  
    * 基於馬可夫決策過程 (Markov Decision Process)。  
  * **自監督式學習 (Self-supervised Learning)**：從無標籤資料中自動生成監督訊號（偽標籤）來訓練模型。  
* **優化演算法 (Optimization Algorithms)**：  
  * 梯度下降 (Gradient Descent)：迭代地調整參數，沿著損失函數梯度的反方向移動，以找到最小值。  
    $ \\mathbf{w}\_{n+1} \= \\mathbf{w}\_n \- \\alpha \\nabla L(\\mathbf{w}\_n) $  
    其中 w 是參數，α 是學習率，L 是損失函數。  
  * 反向傳播 (Backpropagation)：一種高效計算損失函數對網路權重梯度的演算法，利用微積分的鏈式法則。  
    $ \\frac{\\partial L}{\\partial w\_{ij}} \= \\frac{\\partial L}{\\partial a\_j} \\frac{\\partial a\_j}{\\partial z\_j} \\frac{\\partial z\_j}{\\partial w\_{ij}} $ (簡化表示)  
  * Adam 優化器 (Adam Optimizer)：一種更進階、自適應學習率的梯度下降變體，結合了動量和 RMSprop 的優點。  
    $ \\mathbf{m}t \= \\beta\_1 \\mathbf{m}{t-1} \+ (1 \- \\beta\_1) \\mathbf{g}\_t $  
    $ \\mathbf{v}t \= \\beta\_2 \\mathbf{v}{t-1} \+ (1 \- \\beta\_2) \\mathbf{g}\_t^2 $  
    $ \\hat{\\mathbf{m}}\_t \= \\mathbf{m}\_t / (1 \- \\beta\_1^t) $  
    $ \\hat{\\mathbf{v}}\_t \= \\mathbf{v}t / (1 \- \\beta\_2^t) $  
    $ \\mathbf{w}{t+1} \= \\mathbf{w}\_t \- \\alpha \\cdot \\hat{\\mathbf{m}}\_t / (\\sqrt{\\hat{\\mathbf{v}}\_t} \+ \\epsilon) $  
    其中 gt​ 是梯度，mt​ 和 vt​ 是梯度的一階和二階矩估計，β1​,β2​ 是衰減率，α 是學習率，ϵ 是為防止除以零而添加的小常數。  
* **損失函數 (Loss Function)**：衡量模型預測與真實目標之間差異的函數，目標是最小化它。  
  * **均方誤差 (Mean Squared Error, MSE)**：$ L \= \\frac{1}{N} \\sum\_{i=1}^N (y\_i \- \\hat{y}\_i)^2 $  
  * **平均絕對誤差 (Mean Absolute Error, MAE)**：$ L \= \\frac{1}{N} \\sum\_{i=1}^N |y\_i \- \\hat{y}\_i| $  
  * **交叉熵損失 (Cross-Entropy Loss)**：常用於分類問題。

## **II. 機器學習生態系統的擴展**

### **2.1 評估與驗證 (Evaluation & Validation)**

* **定義**：衡量模型性能、可靠性和泛化能力的過程 3。  
* **關鍵指標**：準確度 (Accuracy)、精確度 (Precision)、召回率 (Recall)、F1 分數 (F1 Score)、均方誤差 (MSE)、ROC AUC 等 5。  
* **交叉驗證 (Cross-Validation)**：確保模型避免過擬合或欠擬合。  
* **可解釋性 (Interpretability)**：理解模型預測背後的原因，尤其在複雜模型中（如「黑箱」模型）。

### **2.2 部署與維護 (Deployment & Maintenance)**

* **定義**：將訓練好的模型整合到實際應用中，並持續監控其性能。  
* **模型部署 (Model Deployment)**：將模型打包並使其可供其他應用程式通過 API 等方式使用。  
* **監控與維護 (Monitoring & Maintenance)**：持續追蹤模型在實際環境中的性能，檢測資料漂移、模型退化等問題，並進行再訓練或更新。  
* **硬體加速 (Hardware Acceleration)**：利用專用硬體（如 GPU、TPU、FPGA）來加速模型的訓練和推斷，提高效率。

## **III. 量子機器學習 (Quantum Machine Learning, QML) \- 專業領域**

QML 是一個跨學科領域，它結合了量子計算和機器學習的原理，旨在利用量子現象（如疊加、糾纏和干涉）來增強機器學習的能力，或利用機器學習來解決量子系統中的問題。

### **3.1 QML 類別概述**

1. **量子增強機器學習 (Quantum-Enhanced Machine Learning, QEML)**：  
   * **定義**：利用量子演算法或量子硬體來加速或改進**經典機器學習任務**。  
   * **「量子」在哪**：使用量子位元、量子門、高維希爾伯特空間和量子演算法來處理資料。  
2. **機器學習於量子系統 (Machine Learning of Quantum Systems, MLQS)**：  
   * **定義**：使用**經典機器學習技術**來分析量子資料、模擬量子系統、優化量子硬體或改進量子演算法。  
   * **「量子」在哪**：機器學習模型處理的資料來源於量子實驗或量子模擬，或用於優化量子硬體。  
3. **量子啟發機器學習 (Quantum-Inspired Machine Learning)**：  
   * **定義**：這類方法是**經典演算法**，但其設計或靈感來源於量子力學的原理，通常在經典電腦上運行。

### **3.2 量子資料表示 (Quantum Data Representation)**

在 QML 中，資料的形式會根據其是「量子增強經典 ML」還是「經典 ML 處理量子資料」而有所不同。然而，最終目標都是將資料轉換為量子電腦可以處理的量子態，或將量子測量結果轉換為經典張量。

#### **3.2.1 量子態的理論表示 (Theoretical Representation of Quantum States)**

* 量子位元 (Qubit)：量子資訊的基本單位。一個單量子位元系統的純態可以表示為複數向量：  
  $ |\\psi\\rangle \= \\alpha |0\\rangle \+ \\beta |1\\rangle $  
  其中 $ \\alpha, \\beta \\in \\mathbb{C} $ 且 $ |\\alpha|^2 \+ |\\beta|^2 \= 1 $。  
  對於多個量子位元，系統狀態通過張量積 (Tensor Product) 組合 14。  
* **密度矩陣 (Density Matrix)**：用於描述純態或混合態（量子系統處於多個純態的機率混合）。對於一個 n 量子位元系統，密度矩陣 $ \\rho $ 是一個 2n×2n 的 Hermitian 矩陣，且 $ \\text{Tr}(\\rho) \= 1 $ 30。  
  * 純態：$ \\rho \= |\\psi\\rangle\\langle\\psi| $  
  * 混合態：$ \\rho \= \\sum\_i p\_i |\\psi\_i\\rangle\\langle\\psi\_i| $，其中 $ p\_i $ 是機率。  
* **量子圖態 (Quantum Graph States)**：一種特殊的量子態，用於將數學圖表示為量子電腦上的量子態 31。  
  * 對於無權重圖 G=(V,E)，量子圖態 $ |G\\rangle $ 可以表示為：  
    $ |G\\rangle \= \\prod\_{(u,v)\\in E} U\_z(u,v) |+\\rangle^{\\otimes n} $  
    其中 $ U\_z(u,v) $ 是一個受控 Z 門，在節點 u 和 v 之間建立糾纏 32。  
  * 對於加權圖，可以使用更通用的相位門：  
    $ U\_z(u,v,w) \= e^{-iw\\sigma\_z^u \\sigma\_z^v} $  
    其中 $ \\sigma\_z $ 是 Pauli Z 矩陣，w 是邊的權重 32。  
  * **穩定子形式 (Stabilizer Formalism)**：圖態也可以通過穩定子算符 $ S\_v \= \\sigma\_x^{(v)} \\prod\_{u \\in N(v)} \\sigma\_z^{(u)} $ 來定義 32。

#### **3.2.2 經典資料到量子態的編碼 (Classical-to-Quantum Data Encoding)**

這是 QEML 的關鍵步驟，將經典資料轉換為量子電腦可處理的量子態。

* **基底編碼 (Basis Encoding)**：  
  * **原理**：最直接的方法，將經典二進位資料直接映射到量子位元的計算基底態 $ |0\\rangle $ 和 $ |1\\rangle $。資訊儲存在這些基底態的振幅中。  
  * 數學表示：對於一個 F 位元的經典二進位字串 $ b\_1 b\_2 \\ldots b\_F $，它被映射到一個 F 量子位元的量子態：  
    $ |b\_1 b\_2 \\ldots b\_F\\rangle \= |b\_1\\rangle \\otimes |b\_2\\rangle \\otimes \\ldots \\otimes |b\_F\\rangle $  
  * **適用性**：適用於離散或二進位資料。  
* **角度編碼 (Angle Encoding)**：  
  * **原理**：將經典資料值編碼為量子門的旋轉角度。這利用了量子態在複平面上旋轉時，其測量機率由相位決定的特性。  
  * 數學表示：對於一個經典輸入特徵 xi​，應用一個參數化旋轉門，例如 Ry​(θi​)，其中 $ \\theta\_i \= f(x\_i) $，而 f 是一個將經典值映射到角度的函數。  
    $ |\\psi\\rangle \= R\_y(f(x\_1)) \\otimes R\_y(f(x\_2)) \\otimes \\ldots \\otimes R\_y(f(x\_F)) |0\\rangle^{\\otimes F} $  
    其中 $R\_y(\\theta) \= \\begin{pmatrix} \\cos(\\theta/2) & \-\\sin(\\theta/2) \\ \\sin(\\theta/2) & \\cos(\\theta/2) \\end{pmatrix} $。  
  * **適用性**：常用於將連續變數編碼到參數化量子電路 (PQCs) 中。  
* **振幅編碼 (Amplitude Encoding)**：  
  * **原理**：將經典資料值編碼為量子態的機率振幅。這允許將指數級大的經典資料集壓縮到較少數量的量子位元中。  
  * 數學表示：對於一個 F 維的經典向量 $ \\mathbf{x} \= \[x\_0, x\_1, \\ldots, x\_{F-1}\] $，其中 F=2n，它被映射到一個 n 量子位元的量子態：  
    $ |\\psi\\rangle \= \\sum\_{i=0}^{F-1} x\_i |i\\rangle $  
    前提是向量必須歸一化，即 $ \\sum\_{i=0}^{F-1} |x\_i|^2 \= 1 $。  
  * **優勢**：提供指數級的資料壓縮。  
  * **挑戰**：對於當前量子硬體而言，高效準備所需的量子態仍然具有挑戰性 34。  
* **量子特徵映射 (Quantum Feature Maps, ϕq​)**：  
  * **原理**：一個更廣泛的概念，將經典資料 x 映射到一個高維量子希爾伯特空間 Hq​ 中的量子態 $ |\\phi\_q(x)\\rangle \= U(x)|0\\rangle $ 36。  
  * 量子核方法 (Quantum Kernel Methods)：基於量子特徵映射，通過計算量子態之間的內積來定義核函數。  
    $ K\_q(x\_i, x\_j) \= |\\langle\\phi\_q(x\_i)|\\phi\_q(x\_j)\\rangle|^2 $ 36

    糾纏門（如 CNOT 門）可以創建糾纏增強型量子核，以更好地捕捉特徵之間的複雜關聯 38。  
* **自適應閾值剪枝 (Adaptive Threshold Pruning, ATP)**：一種編碼方法，旨在減少糾纏並優化資料複雜性，以實現 QNN 的高效計算。ATP 動態地剪枝資料中非必要的特徵，以減少量子電路需求，同時保持高性能 39。

#### **3.2.3 量子測量結果的經典表示 (Classical Representation of Quantum Measurement Outcomes)**

當量子資料被經典 ML 模型處理時，它通常以測量結果的形式出現。

* **位元串 (Bitstrings)**：量子測量的結果是機率性的，並產生經典的二進位位元串（0 和 1）。這些位元串可以被經典 ML 模型直接分析。  
* **期望值 (Expectation Values)**：對量子態測量一個可觀測量 $ \\hat{O} $ 的平均結果。  
  * 對於純態 $ |\\psi\\rangle ： \\langle \\hat{O} \\rangle \= \\langle \\psi | \\hat{O} | \\psi \\rangle $ 30  
  * 對於混合態 $ \\rho ： \\langle \\hat{O} \\rangle \= \\text{Tr}(\\hat{O}\\rho) $ 30

    這些期望值是經典的數值，可以作為張量輸入到經典 ML 模型中 17。  
* **頻率/機率分佈 (Frequencies/Probability Distributions)**：重複測量會產生測量結果的頻率分佈，這些頻率可以被視為機率，並作為經典張量輸入。

#### **3.2.4 量子過程的表示 (Representation of Quantum Processes)**

* **量子通道 (Quantum Channels)**：描述量子態如何演化，包括噪聲和操作。它們是完全正且保持跡的線性映射（CPTP maps）40。

  $ \\mathcal{E}(\\rho) \= \\sum\_k E\_k \\rho E\_k^\\dagger $  
  其中 $ E\_k $ 是 Kraus 算符。ML 模型可以學習這些通道的參數來進行噪聲特性化 42。  
* **張量網路噪聲特性化 (Tensor Network Noise Characterization)**：利用張量網路 (TN) 來表示和特性化近程量子電腦上的噪聲通道。這種方法可以高效地從實驗測量中學習噪聲模型，即使對於具有相關噪聲的複雜量子電路也能實現準確估計 43。

### **1.2 模型 (Model)**

模型是學習資料模式的數學函數或計算結構。

* **定義**：一個包含可學習參數的數學函數，用於從輸入資料映射到輸出預測。  
* 神經元 (Neuron)：基本計算單元，執行加權求和、加偏差和激活函數。  
  a=σ(w⋅x+b)  
  其中 x 是輸入向量，w 是權重向量，b 是偏差，σ 是激活函數，a 是輸出。  
* **激活函數 (Activation Function)**：引入非線性，使模型能夠學習複雜的模式。常見的激活函數包括 ReLU、Sigmoid 和 Tanh。  
* **層 (Layer)**：神經元的集合，對輸入進行轉換。  
  * 密集層 (Dense Layer)：每個輸入神經元連接到下一層的所有神經元。  
    $ \\text{output} \= \\text{activation} (\\text{dot}(\\text{input}, \\text{kernel}) \+ \\text{bias}) $  
  * **卷積層 (Convolutional Layer)**：使用濾波器（核）在局部區域內提取特徵。  
  * **循環層 (Recurrent Layer)**：具有內部隱藏狀態，處理序列資料並捕捉時間依賴性。  
  * **注意力層 (Attention Layer)**：動態地為輸入序列的不同部分分配權重，以捕捉上下文關係 26。  
* **模型架構 (Model Architecture)**：層次的組合方式（例如，編碼器-解碼器結構、多頭注意力機制）。

#### **III. 學習機制 (Learning Mechanism)**

這是模型如何從資料中學習並調整參數的過程。

* **定義**：用於調整模型參數以最小化損失函數的演算法。  
* **學習範式**：  
  * **監督式學習 (Supervised Learning)**：使用帶有標籤的資料集訓練模型，以分類或預測結果。  
    * 目標：學習函數 $ y \= f(\\mathbf{x}) $，其中 $y$ 是輸出，$\\mathbf{x}$ 是輸入。  
  * **非監督式學習 (Unsupervised Learning)**：使用無標籤資料集，發現隱藏模式或資料分組。  
    * 目標：學習資料的機率分佈 $ p(\\mathbf{X}) $。  
  * **半監督式學習 (Semi-supervised Learning)**：結合少量標籤資料和大量無標籤資料 1。  
  * **強化學習 (Reinforcement Learning)**：代理通過與環境互動，接收獎勵或懲罰來學習決策策略。  
    * 基於馬可夫決策過程 (Markov Decision Process)。  
  * **自監督式學習 (Self-supervised Learning)**：從無標籤資料中自動生成監督訊號（偽標籤）來訓練模型。  
* **優化演算法 (Optimization Algorithms)**：  
  * 梯度下降 (Gradient Descent)：迭代地調整參數，沿著損失函數梯度的反方向移動，以找到最小值。  
    $ \\mathbf{w}\_{n+1} \= \\mathbf{w}\_n \- \\alpha \\nabla L(\\mathbf{w}\_n) $  
    其中 w 是參數，α 是學習率，L 是損失函數。  
  * 反向傳播 (Backpropagation)：一種高效計算損失函數對網路權重梯度的演算法，利用微積分的鏈式法則。  
    $ \\frac{\\partial L}{\\partial w\_{ij}} \= \\frac{\\partial L}{\\partial a\_j} \\frac{\\partial a\_j}{\\partial z\_j} \\frac{\\partial z\_j}{\\partial w\_{ij}} $ (簡化表示)  
  * Adam 優化器 (Adam Optimizer)：一種更進階、自適應學習率的梯度下降變體，結合了動量和 RMSprop 的優點。  
    $ \\mathbf{m}t \= \\beta\_1 \\mathbf{m}{t-1} \+ (1 \- \\beta\_1) \\mathbf{g}\_t $  
    $ \\mathbf{v}t \= \\beta\_2 \\mathbf{v}{t-1} \+ (1 \- \\beta\_2) \\mathbf{g}\_t^2 $  
    $ \\hat{\\mathbf{m}}\_t \= \\mathbf{m}\_t / (1 \- \\beta\_1^t) $  
    $ \\hat{\\mathbf{v}}\_t \= \\mathbf{v}t / (1 \- \\beta\_2^t) $  
    $ \\mathbf{w}{t+1} \= \\mathbf{w}\_t \- \\alpha \\cdot \\hat{\\mathbf{m}}\_t / (\\sqrt{\\hat{\\mathbf{v}}\_t} \+ \\epsilon) $  
    其中 gt​ 是梯度，mt​ 和 vt​ 是梯度的一階和二階矩估計，β1​,β2​ 是衰減率，α 是學習率，ϵ 是為防止除以零而添加的小常數。  
* **損失函數 (Loss Function)**：衡量模型預測與真實目標之間差異的函數，目標是最小化它。  
  * **均方誤差 (Mean Squared Error, MSE)**：$ L \= \\frac{1}{N} \\sum\_{i=1}^N (y\_i \- \\hat{y}\_i)^2 $  
  * **平均絕對誤差 (Mean Absolute Error, MAE)**：$ L \= \\frac{1}{N} \\sum\_{i=1}^N |y\_i \- \\hat{y}\_i| $  
  * **交叉熵損失 (Cross-Entropy Loss)**：常用於分類問題。

## **II. 機器學習生態系統的擴展**

### **2.1 評估與驗證 (Evaluation & Validation)**

* **定義**：衡量模型性能、可靠性和泛化能力的過程 3。  
* **關鍵指標**：準確度 (Accuracy)、精確度 (Precision)、召回率 (Recall)、F1 分數 (F1 Score)、均方誤差 (MSE)、ROC AUC 等 5。  
* **交叉驗證 (Cross-Validation)**：確保模型避免過擬合或欠擬合。  
* **可解釋性 (Interpretability)**：理解模型預測背後的原因，尤其在複雜模型中（如「黑箱」模型）。

### **2.2 部署與維護 (Deployment & Maintenance)**

* **定義**：將訓練好的模型整合到實際應用中，並持續監控其性能。  
* **模型部署 (Model Deployment)**：將模型打包並使其可供其他應用程式通過 API 等方式使用。  
* **監控與維護 (Monitoring & Maintenance)**：持續追蹤模型在實際環境中的性能，檢測資料漂移、模型退化等問題，並進行再訓練或更新。  
* **硬體加速 (Hardware Acceleration)**：利用專用硬體（如 GPU、TPU、FPGA）來加速模型的訓練和推斷，提高效率。

## **III. 量子機器學習 (Quantum Machine Learning, QML) \- 專業領域**

QML 是一個跨學科領域，它結合了量子計算和機器學習的原理，旨在利用量子現象（如疊加、糾纏和干涉）來增強機器學習的能力，或利用機器學習來解決量子系統中的問題。

### **3.1 QML 類別概述**

1. **量子增強機器學習 (Quantum-Enhanced Machine Learning, QEML)**：  
   * **定義**：利用量子演算法或量子硬體來加速或改進**經典機器學習任務**。  
   * **「量子」在哪**：使用量子位元、量子門、高維希爾伯特空間和量子演算法來處理資料。  
2. **機器學習於量子系統 (Machine Learning of Quantum Systems, MLQS)**：  
   * **定義**：使用**經典機器學習技術**來分析量子資料、模擬量子系統、優化量子硬體或改進量子演算法。  
   * **「量子」在哪**：機器學習模型處理的資料來源於量子實驗或量子模擬，或用於優化量子硬體。  
3. **量子啟發機器學習 (Quantum-Inspired Machine Learning)**：  
   * **定義**：這類方法是**經典演算法**，但其設計或靈感來源於量子力學的原理，通常在經典電腦上運行。

### **3.2 量子資料表示 (Quantum Data Representation)**

在 QML 中，資料的形式會根據其是「量子增強經典 ML」還是「經典 ML 處理量子資料」而有所不同。然而，最終目標都是將資料轉換為量子電腦可以處理的量子態，或將量子測量結果轉換為經典張量。

#### **3.2.1 量子態的理論表示 (Theoretical Representation of Quantum States)**

* 量子位元 (Qubit)：量子資訊的基本單位。一個單量子位元系統的純態可以表示為複數向量：  
  $ |\\psi\\rangle \= \\alpha |0\\rangle \+ \\beta |1\\rangle $  
  其中 $ \\alpha, \\beta \\in \\mathbb{C} $ 且 $ |\\alpha|^2 \+ |\\beta|^2 \= 1 $。  
  對於多個量子位元，系統狀態通過張量積 (Tensor Product) 組合 14。  
* **密度矩陣 (Density Matrix)**：用於描述純態或混合態（量子系統處於多個純態的機率混合）。對於一個 n 量子位元系統，密度矩陣 $ \\rho $ 是一個 2n×2n 的 Hermitian 矩陣，且 $ \\text{Tr}(\\rho) \= 1 $ 30。  
  * 純態：$ \\rho \= |\\psi\\rangle\\langle\\psi| $  
  * 混合態：$ \\rho \= \\sum\_i p\_i |\\psi\_i\\rangle\\langle\\psi\_i| $，其中 $ p\_i $ 是機率。  
* **量子圖態 (Quantum Graph States)**：一種特殊的量子態，用於將數學圖表示為量子電腦上的量子態 31。  
  * 對於無權重圖 G=(V,E)，量子圖態 $ |G\\rangle $ 可以表示為：  
    $ |G\\rangle \= \\prod\_{(u,v)\\in E} U\_z(u,v) |+\\rangle^{\\otimes n} $  
    其中 $ U\_z(u,v) $ 是一個受控 Z 門，在節點 u 和 v 之間建立糾纏 32。  
  * 對於加權圖，可以使用更通用的相位門：  
    $ U\_z(u,v,w) \= e^{-iw\\sigma\_z^u \\sigma\_z^v} $  
    其中 $ \\sigma\_z $ 是 Pauli Z 矩陣，w 是邊的權重 32。  
  * **穩定子形式 (Stabilizer Formalism)**：圖態也可以通過穩定子算符 $ S\_v \= \\sigma\_x^{(v)} \\prod\_{u \\in N(v)} \\sigma\_z^{(u)} $ 來定義 32。

#### **3.2.2 經典資料到量子態的編碼 (Classical-to-Quantum Data Encoding)**

這是 QEML 的關鍵步驟，將經典資料轉換為量子電腦可處理的量子態。

* **基底編碼 (Basis Encoding)**：  
  * **原理**：最直接的方法，將經典二進位資料直接映射到量子位元的計算基底態 $ |0\\rangle $ 和 $ |1\\rangle $。資訊儲存在這些基底態的振幅中。  
  * 數學表示：對於一個 F 位元的經典二進位字串 $ b\_1 b\_2 \\ldots b\_F $，它被映射到一個 F 量子位元的量子態：  
    $ |b\_1 b\_2 \\ldots b\_F\\rangle \= |b\_1\\rangle \\otimes |b\_2\\rangle \\otimes \\ldots \\otimes |b\_F\\rangle $  
  * **適用性**：適用於離散或二進位資料。  
* **角度編碼 (Angle Encoding)**：  
  * **原理**：將經典資料值編碼為量子門的旋轉角度。這利用了量子態在複平面上旋轉時，其測量機率由相位決定的特性。  
  * 數學表示：對於一個經典輸入特徵 xi​，應用一個參數化旋轉門，例如 Ry​(θi​)，其中 $ \\theta\_i \= f(x\_i) $，而 f 是一個將經典值映射到角度的函數。  
    $ |\\psi\\rangle \= R\_y(f(x\_1)) \\otimes R\_y(f(x\_2)) \\otimes \\ldots \\otimes R\_y(f(x\_F)) |0\\rangle^{\\otimes F} $  
    其中 $R\_y(\\theta) \= \\begin{pmatrix} \\cos(\\theta/2) & \-\\sin(\\theta/2) \\ \\sin(\\theta/2) & \\cos(\\theta/2) \\end{pmatrix} $。  
  * **適用性**：常用於將連續變數編碼到參數化量子電路 (PQCs) 中。  
* **振幅編碼 (Amplitude Encoding)**：  
  * **原理**：將經典資料值編碼為量子態的機率振幅。這允許將指數級大的經典資料集壓縮到較少數量的量子位元中。  
  * 數學表示：對於一個 F 維的經典向量 $ \\mathbf{x} \= \[x\_0, x\_1, \\ldots, x\_{F-1}\] $，其中 F=2n，它被映射到一個 n 量子位元的量子態：  
    $ |\\psi\\rangle \= \\sum\_{i=0}^{F-1} x\_i |i\\rangle $  
    前提是向量必須歸一化，即 $ \\sum\_{i=0}^{F-1} |x\_i|^2 \= 1 $。  
  * **優勢**：提供指數級的資料壓縮。  
  * **挑戰**：對於當前量子硬體而言，高效準備所需的量子態仍然具有挑戰性 34。  
* **量子特徵映射 (Quantum Feature Maps, ϕq​)**：  
  * **原理**：一個更廣泛的概念，將經典資料 x 映射到一個高維量子希爾伯特空間 Hq​ 中的量子態 $ |\\phi\_q(x)\\rangle \= U(x)|0\\rangle $ 36。  
  * 量子核方法 (Quantum Kernel Methods)：基於量子特徵映射，通過計算量子態之間的內積來定義核函數。  
    $ K\_q(x\_i, x\_j

#### **引用的著作**

1. Quantum Machine Learning 101: Beginner's Guide to Big Data \- BlueQubit, 檢索日期：8月 2, 2025， [https://www.bluequbit.io/quantum-machine-learning](https://www.bluequbit.io/quantum-machine-learning)  
2. What Is Machine Learning (ML)? \- IBM, 檢索日期：8月 2, 2025， [https://www.ibm.com/think/topics/machine-learning](https://www.ibm.com/think/topics/machine-learning)  
3. Machine Learning Components: Elements & Classifications \- lakeFS, 檢索日期：8月 2, 2025， [https://lakefs.io/blog/machine-learning-components/](https://lakefs.io/blog/machine-learning-components/)  
4. (PDF) Enhancing Machine Learning Workflows: A Comprehensive Study of Machine Learning Pipelines \- ResearchGate, 檢索日期：8月 2, 2025， [https://www.researchgate.net/publication/379431932\_Enhancing\_Machine\_Learning\_Workflows\_A\_Comprehensive\_Study\_of\_Machine\_Learning\_Pipelines](https://www.researchgate.net/publication/379431932_Enhancing_Machine_Learning_Workflows_A_Comprehensive_Study_of_Machine_Learning_Pipelines)  
5. a comparative analysis of classical-to-quantum mapping techniques ..., 檢索日期：8月 2, 2025， [https://d-nb.info/1353471489/34](https://d-nb.info/1353471489/34)  
6. Linear regression: Loss | Machine Learning \- Google for Developers, 檢索日期：8月 4, 2025， [https://developers.google.com/machine-learning/crash-course/linear-regression/loss](https://developers.google.com/machine-learning/crash-course/linear-regression/loss)  
7. Understanding Message Passing in Graph Neural Networks | by ..., 檢索日期：8月 2, 2025， [https://medium.com/@rajeev.chandran\_61731/understanding-message-passing-frameworks-in-graph-neural-networks-944d9e2a1105](https://medium.com/@rajeev.chandran_61731/understanding-message-passing-frameworks-in-graph-neural-networks-944d9e2a1105)  
8. Graph Neural Network: In a Nutshell | Karthick Panner Selvam, 檢索日期：8月 2, 2025， [https://karthick.ai/blog/2024/Graph-Neural-Network/](https://karthick.ai/blog/2024/Graph-Neural-Network/)  
9. Data Requirements for Machine Learning \- TDWI, 檢索日期：8月 2, 2025， [https://tdwi.org/articles/2018/09/14/adv-all-data-requirements-for-machine-learning.aspx](https://tdwi.org/articles/2018/09/14/adv-all-data-requirements-for-machine-learning.aspx)  
10. How Much Data Is Required for Machine Learning? \- PostIndustria, 檢索日期：8月 2, 2025， [https://postindustria.com/how-much-data-is-required-for-machine-learning/](https://postindustria.com/how-much-data-is-required-for-machine-learning/)  
11. From Graphs to Qubits: A Critical Review of Quantum Graph Neural Networks \- arXiv, 檢索日期：8月 2, 2025， [https://arxiv.org/html/2408.06524](https://arxiv.org/html/2408.06524)  
12. DuoGNN: Topology-aware Graph Neural Network with Homophily and Heterophily Interaction-Decoupling \- arXiv, 檢索日期：8月 2, 2025， [https://arxiv.org/html/2409.19616v1](https://arxiv.org/html/2409.19616v1)  
13. A unifying primary framework for quantum graph neural ... \- arXiv, 檢索日期：8月 2, 2025， [https://arxiv.org/abs/2402.13001](https://arxiv.org/abs/2402.13001)  
14. arxiv.org, 檢索日期：8月 2, 2025， [https://arxiv.org/html/2412.19211v1](https://arxiv.org/html/2412.19211v1)  
15. Challenges and Opportunities of Quantum Machine Learning, 檢索日期：8月 2, 2025， [https://www.icvtank.com/newsinfo/828689.html?templateId=287088](https://www.icvtank.com/newsinfo/828689.html?templateId=287088)  
16. Transformer (deep learning architecture) \- Wikipedia, 檢索日期：8月 2, 2025， [https://en.wikipedia.org/wiki/Transformer\_(deep\_learning\_architecture)](https://en.wikipedia.org/wiki/Transformer_\(deep_learning_architecture\))  
17. arXiv:2401.12007v3 \[cs.LG\] 30 Jan 2024, 檢索日期：8月 2, 2025， [https://arxiv.org/pdf/2401.12007](https://arxiv.org/pdf/2401.12007)  
18. Quantum Machine Learning and Data Re-Uploading: Evaluation on Benchmark and Laboratory Medicine Datasets | medRxiv, 檢索日期：8月 2, 2025， [https://www.medrxiv.org/content/10.1101/2025.05.14.25327605v1](https://www.medrxiv.org/content/10.1101/2025.05.14.25327605v1)  
19. \[2402.11014\] Neural-network quantum states for many-body physics \- arXiv, 檢索日期：8月 2, 2025， [https://arxiv.org/abs/2402.11014](https://arxiv.org/abs/2402.11014)  
20. Quantum phase detection generalization from marginal quantum neural network models | Phys. Rev. B \- Physical Review Link Manager, 檢索日期：8月 2, 2025， [https://link.aps.org/doi/10.1103/PhysRevB.107.L081105](https://link.aps.org/doi/10.1103/PhysRevB.107.L081105)  
21. \[2503.19476\] Extracting Interpretable Logic Rules from Graph Neural Networks \- arXiv, 檢索日期：8月 2, 2025， [https://arxiv.org/abs/2503.19476](https://arxiv.org/abs/2503.19476)  
22. Generative quantum combinatorial optimization by means of a novel conditional generative quantum eigensolver \- arXiv, 檢索日期：8月 2, 2025， [https://arxiv.org/html/2501.16986v1](https://arxiv.org/html/2501.16986v1)  
23. Quantum generative modeling for financial time series with ... \- arXiv, 檢索日期：8月 2, 2025， [https://arxiv.org/abs/2507.22035](https://arxiv.org/abs/2507.22035)  
24. arxiv.org, 檢索日期：8月 2, 2025， [https://arxiv.org/html/2504.00034v1](https://arxiv.org/html/2504.00034v1)  
25. \[2502.19970\] Quantum generative classification with mixed states \- arXiv, 檢索日期：8月 2, 2025， [https://arxiv.org/abs/2502.19970](https://arxiv.org/abs/2502.19970)  
26. Attention (machine learning) \- Wikipedia, 檢索日期：8月 2, 2025， [https://en.wikipedia.org/wiki/Attention\_(machine\_learning)](https://en.wikipedia.org/wiki/Attention_\(machine_learning\))  
27. Multi-Head Attention Mechanism \- GeeksforGeeks, 檢索日期：8月 2, 2025， [https://www.geeksforgeeks.org/nlp/multi-head-attention-mechanism/](https://www.geeksforgeeks.org/nlp/multi-head-attention-mechanism/)  
28. What is an attention mechanism? | IBM, 檢索日期：8月 2, 2025， [https://www.ibm.com/think/topics/attention-mechanism](https://www.ibm.com/think/topics/attention-mechanism)  
29. Are LLMs just predicting the next token? : r/ArtificialInteligence \- Reddit, 檢索日期：8月 2, 2025， [https://www.reddit.com/r/ArtificialInteligence/comments/1jo3o69/are\_llms\_just\_predicting\_the\_next\_token/](https://www.reddit.com/r/ArtificialInteligence/comments/1jo3o69/are_llms_just_predicting_the_next_token/)  
30. arxiv.org, 檢索日期：8月 2, 2025， [https://arxiv.org/abs/2502.09488](https://arxiv.org/abs/2502.09488)  
31. Inductive Graph Representation Learning with Quantum Graph Neural Networks \- arXiv, 檢索日期：8月 2, 2025， [https://arxiv.org/pdf/2503.24111](https://arxiv.org/pdf/2503.24111)  
32. A unifying primary framework for quantum graph neural ... \- arXiv, 檢索日期：8月 2, 2025， [https://arxiv.org/pdf/2402.13001](https://arxiv.org/pdf/2402.13001)  
33. \[2501.06002\] DeltaGNN: Graph Neural Network with Information Flow Control \- arXiv, 檢索日期：8月 2, 2025， [https://arxiv.org/abs/2501.06002](https://arxiv.org/abs/2501.06002)  
34. Predicting the von Neumann Entanglement Entropy Using a Graph Neural Network \- arXiv, 檢索日期：8月 2, 2025， [https://arxiv.org/html/2503.23635v2](https://arxiv.org/html/2503.23635v2)  
35. Faithful novel machine learning for predicting quantum properties \- ResearchGate, 檢索日期：8月 2, 2025， [https://www.researchgate.net/publication/394034259\_Faithful\_novel\_machine\_learning\_for\_predicting\_quantum\_properties](https://www.researchgate.net/publication/394034259_Faithful_novel_machine_learning_for_predicting_quantum_properties)  
36. Towards Quantum Graph Neural Networks: An Ego-Graph Learning Approach \- arXiv, 檢索日期：8月 2, 2025， [https://arxiv.org/html/2201.05158v3](https://arxiv.org/html/2201.05158v3)  
37. Transformer Model: The Basics and 7 Models You Should Know, 檢索日期：8月 2, 2025， [https://swimm.io/learn/large-language-models/transformer-model-the-basics-and-7-models-you-should-know](https://swimm.io/learn/large-language-models/transformer-model-the-basics-and-7-models-you-should-know)  
38. (PDF) Can Entanglement-enhanced Quantum Kernels Improve Data ..., 檢索日期：8月 2, 2025， [https://www.researchgate.net/publication/381157966\_Can\_Entanglement-enhanced\_Quantum\_Kernels\_Improve\_Data\_Classification](https://www.researchgate.net/publication/381157966_Can_Entanglement-enhanced_Quantum_Kernels_Improve_Data_Classification)  
39. arXiv:2503.21815v1 \[quant-ph\] 26 Mar 2025, 檢索日期：8月 2, 2025， [https://arxiv.org/pdf/2503.21815?](https://arxiv.org/pdf/2503.21815)  
40. Machine Learning of Noise-Resilient Quantum Circuits \- Physical Review Link Manager, 檢索日期：8月 2, 2025， [https://link.aps.org/doi/10.1103/PRXQuantum.2.010324](https://link.aps.org/doi/10.1103/PRXQuantum.2.010324)  
41. Variational optimization of the amplitude of neural-network quantum ..., 檢索日期：8月 2, 2025， [https://link.aps.org/doi/10.1103/PhysRevB.109.245120](https://link.aps.org/doi/10.1103/PhysRevB.109.245120)  
42. Noisy Quantum Channel Characterization Using Quantum Neural ..., 檢索日期：8月 2, 2025， [https://www.mdpi.com/2079-9292/12/11/2430](https://www.mdpi.com/2079-9292/12/11/2430)  
43. arXiv:2402.08556v2 \[quant-ph\] 2 Sep 2024, 檢索日期：8月 2, 2025， [https://arxiv.org/pdf/2402.08556](https://arxiv.org/pdf/2402.08556)  
44. \[2402.08556\] Tensor network noise characterization for near-term quantum computers, 檢索日期：8月 2, 2025， [https://arxiv.org/abs/2402.08556](https://arxiv.org/abs/2402.08556)