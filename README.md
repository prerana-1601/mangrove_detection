# 🛰️ Mangrove Segmentation using U-Net

This project implements a semantic segmentation pipeline to accurately identify **mangrove forests** from satellite imagery using a **U-Net architecture**. The dataset is curated from Google Earth Engine (GEE) and enriched with vegetation and elevation features to generate accurate binary masks.

---

## 🧠 Project Highlights

- 🔍 **Remote sensing-based semantic segmentation**
- 🌿 **NDVI & elevation-based mask generation**
- 🧱 **U-Net model** trained on image patches
- 🧪 Custom metrics: **IoU**, **Dice**, **Precision**, and more
- 🧰 Modular code for easy patch generation, training, prediction, and evaluation


---

## 🛰️ Data Pipeline

1. **Satellite Imagery**: Downloaded using GEE (Sentinel/Landsat)
2. **Binary Mask Generation**: Using NDVI & elevation thresholds
3. **Patch Creation**: 128x128 image-mask pairs for training
4. **Augmentation**: Flipping, rotation, and contrast variation
5. **Model Training**: U-Net trained with weighted BCE + Jaccard Loss
6. **Prediction**: Patch-wise prediction combined into full-scale mask

---

## 🧠 Model Architecture

The model follows a standard [U-Net](https://arxiv.org/abs/1505.04597) architecture with:
- Contracting path: Convolutions + MaxPooling
- Bottleneck
- Expanding path: Up-convolutions + Skip connections

Final activation: `Sigmoid` (for binary mask output)

---

## 📊 Evaluation Metrics

- **IoU (Jaccard Index)**
- **Dice Coefficient**
- **Pixel Accuracy**
- **Precision / Recall**

---

