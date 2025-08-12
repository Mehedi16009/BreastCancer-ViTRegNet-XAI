# BreastCancer-ViTRegNet-XAI
A hybrid ViT + RegNet-based breast cancer classification pipeline with integrated GradCAM and SHAP explainability, trained on the CBIS-DDSM mammogram dataset.


## 🩺 Breast Cancer Diagnosis using ViT + RegNet with Explainability

This repository contains a complete deep learning pipeline for **breast cancer classification** using **Vision Transformers (ViT)** and **RegNetY** architectures, enhanced with **GradCAM** and **SHAP** for model interpretability. The models are trained and evaluated on the **CBIS-DDSM mammography dataset**.

---

## 🔍 Project Highlights

- ✅ Image classification (benign vs malignant) using **ViT** and **RegNetY**
- ✅ Hybrid **fusion model** combining ViT and RegNet feature maps
- ✅ Model **explainability** with **GradCAM** and **Captum SHAP**
- ✅ High-quality visualizations (CAM overlays, SHAP maps)
- ✅ Built using **PyTorch**, **PyTorch Lightning**, and **timm**

---

## 📁 Dataset

- 📦 Dataset: [CBIS-DDSM Kaggle Link]([https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset?resource=download))
- 📄 Metadata: Includes `mass_case_description_train_set.csv`, `calc_case_description_train_set.csv`, and `meta.csv`
- 🖼️ Images: 6774 patient JPEG mammogram folders

---
##🚀 Project Structure
├── dataset/
│   ├── mass_case_description_train_set.csv
│   ├── calc_case_description_train_set.csv
│   ├── meta.csv
│   └── images/
├── models/
│   ├── vit_classifier.py
│   ├── regnet_classifier.py
│   └── fusion_model.py
├── explainability/
│   ├── gradcam_vit.py
│   ├── gradcam_regnet.py
│   └── shap_explainer.py
├── notebooks/
│   ├── training_pipeline.ipynb
│   └── evaluation_and_visuals.ipynb
├── utils/
│   ├── dataset_loader.py
│   └── transforms.py
├── main.py
├── README.md
└── requirements.txt

## Output Images:
Precision-Recall Curve:
<img width="700" height="600" alt="precision_recall_curve" src="https://github.com/user-attachments/assets/e9cf4a1c-e329-426b-961b-e418950e561e" />


Loss Curve:
<img width="700" height="600" alt="loss_curve" src="https://github.com/user-attachments/assets/6b6c4e0b-0d93-42d5-817f-683bc9ac429f" />


F1-Curve:
<img width="700" height="600" alt="f1_curve" src="https://github.com/user-attachments/assets/30fa0c79-549d-4fe4-bb22-732cedfb3f3e" />


Accuracy Curve:
<img width="700" height="600" alt="accuracy_curve" src="https://github.com/user-attachments/assets/52172710-0395-487b-bde4-53463e3bbb5f" />




## 📦 Requirements
```bash
- Python 3.8+
- PyTorch ≥ 1.13
- PyTorch Lightning
- timm
- scikit-learn
- captum
- pytorch-grad-cam
- albumentations
- matplotlib
- opencv-python
```
Install all packages:
```bash
pip install -r requirements.txt
```
##📬 Contact:
Developed by Md Mehedi Hasan
Email: mehedi.hasan.ict@mbstu.ac.bd
Institution: [GIIT University / IdeaVerse / MBSTU]
