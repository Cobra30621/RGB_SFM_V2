
# 🧠 A New CNN-Based Interpretable Deep Learning Model

[📄 Paper (Master Thesis)](https://etd.lib.nycu.edu.tw/cgi-bin/gs32/ncugsweb.cgi?o=dncucdr&s=id=%22GC111522094%22.&searchmode=basic)  
[📊 Introduction Slides (Google Drive)](https://docs.google.com/presentation/d/1RVQyYK1Z_ld_ynAwWAJSIH9tQb7rh1GJ/edit?usp=drive_link&ouid=115867738728025033314&rtpof=true&sd=true)

---

## 📌 簡介

本研究提出一個基於卷積神經網路（CNN）的新型可解釋性深度學習模型，整體架構可分為三個模組：

- **色彩感知模組**：計算輸入影像與 30 種基礎顏色的相似度，提取色彩特徵。
- **輪廓感知模組**：將影像轉為灰階後，使用高斯卷積與特徵增強提取輪廓資訊。
- **特徵傳遞模組**：合併來自色彩與輪廓模組的資訊，透過多層高斯卷積傳遞至最終分類層。

### 🪄 可解釋性方法

本模型可產生：
1. 每層 filter 的對應圖形。
2. 對每張輸入圖片，選出每層最有反應的特徵響應圖。
3. 合成不同層的視覺化結果，使用者可觀察模型的決策依據。

---

## 🧱 模型架構

### 整體結構圖
<img src="https://github.com/user-attachments/assets/092c883e-4de1-4b1b-af0c-a31526849c7f" width="800">

### 子模組結構
<img src="https://github.com/user-attachments/assets/6e3198eb-b0eb-43ec-b479-a46fc0cecdb8" width="600">

---

## 👁 可解釋性原理示意

<img src="https://github.com/user-attachments/assets/a2300822-8adf-45db-92aa-45957f2d44b5" width="600">
<img src="https://github.com/user-attachments/assets/c0fc6de0-89f3-4c9d-bbae-51b0c5d113c4" width="600">
<img src="https://github.com/user-attachments/assets/9330de2e-35d3-4fe1-a63c-302f7f3a780b" width="600">
<img src="https://github.com/user-attachments/assets/b4f46300-ff98-4bae-8b1a-6be2eb2fad35" width="600">

---

## 🖼 可解釋性成果範例

<img src="https://github.com/user-attachments/assets/49552508-29ea-4584-98ac-0d357f2d0d7f" width="600">

---

## ⚙️ 安裝說明

```bash
# 1. 建立 Conda 環境
conda create --name SFM python=3.9
conda activate SFM

# 2. 安裝 PyTorch（請依 GPU 支援下載對應版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 安裝其他套件
pip install -r requirements.txt
```

---

## 📦 資料集準備

請至實驗室 NAS 下載資料集，並建立 `data/` 資料夾將其放入。

---

## 🚀 執行專案

### 模型訓練

請於 `config.py` 中設定訓練參數後執行
config 可參考 [config_example.py](config_example.py)，具體參數涵義參考 [config.md](config.md)

```bash
# 一般訓練
python train.py

# K-Fold 訓練
python train_kfold.py
```

### 畫圖功能

```bash
# 可解釋性視覺化
python plot_example_V2.py

# CI (Critical Inputs) 圖
python plot_CI_V2.py

# 指標統計圖
python plot_stats_metrics.py

# 一次畫全部
python plot_every_graph.py
```

### GUI 介面（操作與展示用）

```bash
python display_gui.py
```

---

## 📁 專案資料夾結構與說明

| 資料夾 / 檔案 | 說明 |
|----------------|------|
| `runs/` | 訓練過程的儲存資料夾（模型權重等） |
| `data/` | 資料集位置 |
| `dataloader/` | 資料載入模組 |
| └── `get_dataloader.py` | 設定自訂資料集來源 |
| `detect/` | 可解釋性圖像儲存處 |
| `loss/` | 自訂 Loss 函式 |
| `model/` | 模型架構定義 |
| ├── `SFMCNN.py` | 2023 景豐版本 |
| ├── `RGB_SFMCNN.py` | 2024 建名版本 |
| └── `RGB_SFMCNN_V2.py` | 2025 俊宇版本 |
| `monitor/` | 濾波器指標監控模組 |
| ├── `metrics.py` | 設定濾波器評估指標 |
| ├── `calculate_status.py` | 統計計算 |
| ├── `monitor_method.py` | 評估流程 |
| ├── `plot_df.py` | 表格視覺化 |
| └── `plot_distribution.py` | 分布圖視覺化 |
| `pth/` | 儲存訓練完成模型的 `.pth` 檔案 |
| `research/` | 研究中使用的輔助程式碼 |

---

## 📚 Citation

```bibtex
@mastersthesis{TU2024InterpretableModel,
  title={以卷積神經網路為基礎之新型可解釋性深度學習模型},
  author={TU, CHIEN-MING and Su, Mu-Chun},
  school={National Central University},
  year={2024}
}
```
