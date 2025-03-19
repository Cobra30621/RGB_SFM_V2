# A New CNN-Based Interpretable Deep Learning Model
Implementation of paper - [https://etd.lib.nycu.edu.tw/cgi-bin/gs32/ncugsweb.cgi/ccd=koGVez/record?r1=1&h1=0](https://etd.lib.nycu.edu.tw/cgi-bin/gs32/ncugsweb.cgi?o=dncucdr&s=id=%22GC111522094%22.&searchmode=basic)
Introduction PPT：

## 簡介
本研究提出了基於卷積神經網路的新型可解釋性深度學習模型，
該模型包括色彩感知區塊、輪廓感知區塊和特徵傳遞區塊三大部分。
色彩感知區塊透過計算輸入影像不同部分的平均色彩與30種基礎色彩的相似度來提取輸入影像的顏色特徵，
輪廓感知區塊則透過前處理將彩色影像變成灰階影像並輸入高斯卷積與特徵增強來檢測影像中的輪廓特徵，
特徵傳遞區塊則將輸入特徵進行高斯卷積與特徵增強後並且將
輸入特徵透過時序性合併的方式組成更完整的特徵輸出到下一層直到傳遞至全連接層，
最後將輸出的色彩特徵與輪廓特徵結合後輸入進全連接層進行分類。 \
\
可解釋性的部分，
論文先產生每一層Filter之特徵響應圖的對應圖形，
之後根據輸入圖片於每一層的輸出值來選出該圖片不同部分最有反應之特徵響應圖的對應圖形，
最終組合出模型針對輸入圖片不同部分在每一層最有反應的特徵的視覺化結果。
使用者可藉由每層的視覺化結果來了解模型在每一層是根據圖形的甚麼特徵來做出最後的判斷。

# 模型架構
模型架構：\
<img src="https://github.com/user-attachments/assets/092c883e-4de1-4b1b-af0c-a31526849c7f" alt="Editor" width="800"> \
各個區塊架構：\
<img src="https://github.com/user-attachments/assets/6e3198eb-b0eb-43ec-b479-a46fc0cecdb8" alt="Editor" width="600">

# 可解釋性原理
<img src="https://github.com/user-attachments/assets/a2300822-8adf-45db-92aa-45957f2d44b5" alt="Editor" width="600">
<img src="https://github.com/user-attachments/assets/c0fc6de0-89f3-4c9d-bbae-51b0c5d113c4" alt="Editor" width="600">
<img src="https://github.com/user-attachments/assets/9330de2e-35d3-4fe1-a63c-302f7f3a780b" alt="Editor" width="600">
<img src="https://github.com/user-attachments/assets/b4f46300-ff98-4bae-8b1a-6be2eb2fad35" alt="Editor" width="600">

# 可解釋性成果
<img src="https://github.com/user-attachments/assets/49552508-29ea-4584-98ac-0d357f2d0d7f" alt="Editor" width="600">


## Installation
```
  1. conda create --name SFM python=3.9
  2. conda activate SFM
  3. Install pytorch==2.0.0 && torchvision==0.15.2 
    - conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
  4. conda install requirements.txt
```

## Train
```
  python train.py
```

## Test
```
  python test.py
```

## Citation
```
  @mastersthesis{TU2024InterpretableModel,
    title={以卷積神經網路為基礎之新型可解釋性深度學習模型},
    author={TU, CHIEN-MING and Su, Mu-Chun},
    school={National Central University},
    year={2024}
  }
```
