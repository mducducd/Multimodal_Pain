# Visual-Physiological Pain Assessment  
**Representation Learning and Dual Attention Fusion**

> ⚠️Disclaimer: This repo is a work in progress and may not run flawlessly out-of-the-box. We're sharing the code as-is for reference and reproducibility.

## 📚 Overview

This repository focuses on **multimodal pain assessment** using both **facial video** and **bio-physiological signals**. The architecture consists of two main stages:

1. **Representation Learning**
   - **Video Branch**: Masked Autoencoder for facial expression modeling, based on [MARLIN](https://github.com/ControlNet/MARLIN).
   - **Signal Branch**: Masked Autoencoder for multivariate time-series signals, based on [MVTS Transformer](https://github.com/gzerveas/mvts_transformer).

2. **Classifier Training**
   - Attention-based fusion of video and signal features for pain-level classification.

---

## 🛠️ Installation

```bash
conda env create -f environment.yml
```
⚙️ The model was trained and evaluated on NVIDIA RTX 8000 GPUs. Video encoder fits within a 16 GB GPU VRAM budget during training with small batch sizes.

## 📁 Dataset
Publicly available third-party datasets:

BioVid Heat Pain Database: Available upon request from the official website [https://www.nit.ovgu.de/BioVid.html](https://www.nit.ovgu.de/BioVid.html, as described in [10.1109/CYBConf.2013.6617456](10.1109/CYBConf.2013.6617456)

AI4Pain dataset: Available upon request from the official website [AI4Pain Challenge](https://sites.google.com/view/ai4pain/challenge-details, as described in [10.1109/ACIIW63320.2024.00012](10.1109/ACIIW63320.2024.00012)

To adapt your own dataset as well as for shape/format changes:
- Modify `model/classifier.py` for architecture.
- Modify dataloaders in `src/dataset/celebv_hq.py`.
- Following Pytorch  [Lightning 1.7.7](https://lightning.ai/docs/pytorch/1.7.7/) modules.


## 🧠 Model Components

| Component        | Description                         | Path |
|------------------|-------------------------------------|------|
| Video MAE        | Visual encoder                      | [`src/marlin_pytorch/model`](https://github.com/mducducd/Multimodal_Pain/tree/main/src/marlin_pytorch/model) |
| Signal MAE       | Time-series encoder                 | [`mvts_transformer/src/models`](https://github.com/mducducd/Multimodal_Pain/tree/main/mvts_transformer/src/models) |
| Cross Attention  | Fusion between modalities           | [`model/crossatten.py`](model/crossatten.py) |
| Classifier       | Final classification head           | [`model/classifier.py`](model/classifier.py) |

## ✅ 🧪 Pre-training
### 🎥 Video Branch

#### 1. Face Preprocessing
```bash
python preprocess/celebvhq_preprocess.py --data_dir /path/to/videos
```

#### 2. Generate masks:
```bash
python preprocess/ytf_preprocess.py --data_dir
```

train_set.csv example for pre-training (from [YoutubeFaces](https://www.cs.tau.ac.il/~wolf/ytfaces/))
```
path,len
AJ_Cook/0,79
AJ_Cook/2,194
Aaron_Sorkin/0,70
Aaron_Sorkin/3,174
Aaron_Tippin/0,119
Aaron_Tippin/1,83
Abdel_Aziz_Al-Hakim/0,103
Abdel_Aziz_Al-Hakim/1,285
Abdel_Aziz_Al-Hakim/4,141
Abdul_Majeed_Shobokshi/1,624
Abdulaziz_Kamilov/4,195
```

#### 3. Video pre-training:

Directory for .csv
```
├── Train
│   ├── cropped
│   │   ├── id
│   │   ├── id
│   │   ├── ...
│   ├── face_parsing_images_DB
│   ├── train.txt
│   ├── val.txt
│   ├── ...
```

```bash
python train.py \
    --config config/pretrain/marlin_vit_base.yaml \
    --data_dir /path/to/youtube_faces \
    --n_gpus 4 \
    --num_workers 8 \
    --batch_size 16 \
    --epochs 2000 \
    --official_pretrained /path/to/checkpoint.pth
```

🧬 Signal Branch

Directory for .csv
```
├── Data
│   ├── id1
│   │   |   ├── video1
│   │   |   |   ├── frame1.jpg
│   │   |   |   ├── ...
│   │   |   ├── ...
│   │   ├── ...
│   ├── id2
│   ├── ...
```

Signal pre-training
```bash
cd mvts_transformer
python src/main.py --output_dir experiments --comment "pretraining through imputation" --name $1_pretrained --records_file Imputation_records.xls --data_dir /path/to/$1/ --data_class pain --pattern TRAIN --val_ratio 0.2 --epochs 700 --lr 0.001 --optimizer RAdam --batch_size 32 --pos_encoding learnable --d_model 128
```

## 🧪 Probing / Classifier Training

Prepare your pre-trained model from previous pre-training, and modify `model/classifier.py` to load them.

Directory Layout:
```
├── Train
│   ├── video
│   │   ├── id
│   │   |   ├── video1
│   │   |   |   ├── frame1.jpg
│   │   |   |   ├── ...
│   │   |   ├── ...
│   │   ├── ...
│   ├── biosignals_filtered
│   │   ├── id
            ├── 1.csv
            ├── ...
│   │   ├── ...
│   ├── celebvhq_info.json
│   ├── train.txt
│   ├── val.txt
│   ├── ...
```

Run Evaluation / Classification

```bash
python3 evaluate.py
```


