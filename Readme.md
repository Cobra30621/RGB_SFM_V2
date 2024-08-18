# RGB_SFMCNN
Implementation of paper - [https://etd.lib.nycu.edu.tw/cgi-bin/gs32/ncugsweb.cgi/ccd=koGVez/record?r1=1&h1=0](https://etd.lib.nycu.edu.tw/cgi-bin/gs32/ncugsweb.cgi?o=dncucdr&s=id=%22GC111522094%22.&searchmode=basic)

## 簡介
本研究提出了基於卷積神經網路的新型可解釋性深度學習模型，
該模型包括色彩感知區塊、輪廓感知區塊和特徵傳遞區塊三大部分。
色彩感知區塊透過計算輸入影像不同部分的平均色彩與30種基礎色彩的相似度來提取輸入影像的顏色特徵，
輪廓感知區塊則透過前處理將彩色影像變成灰階影像並輸入高斯卷積與特徵增強來檢測影像中的輪廓特徵，
特徵傳遞區塊則將輸入特徵進行高斯卷積與特徵增強後並且將
輸入特徵透過時序性合併的方式組成更完整的特徵輸出到下一層直到傳遞至全連接層，
最後將輸出的色彩特徵與輪廓特徵結合後輸入進全連接層進行分類。
![image](https://github.com/user-attachments/assets/73ed409d-ceb6-4bf8-a830-622d35f9d0da)

## Installation
```
  1. conda create --name SFM python=3.9
  2. conda activate SFM
  3. Install pytorch==2.0. && torchvision==0.15.2
  4. conda install requirements.txt
```

## Train
```
  python train.py
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
