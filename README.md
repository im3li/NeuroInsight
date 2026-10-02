# NeuroInsight

**An Explainable Multi-Dataset Decision Support Framework for Alzheimer's Disease Classification Using MRI Images**

---

## Overview

NeuroInsight is an AI-powered decision support prototype that classifies brain MRI scans into Alzheimer's disease stages. Unlike conventional classifiers, NeuroInsight combines multiple AI techniques into a single end-to-end pipeline — delivering not just a prediction, but also visual explanations, confidence scores, and a clinician-friendly report.

The system is designed to demonstrate how explainable AI can support transparent and reliable Alzheimer's screening.

---

## Key Features

- **Multi-Dataset Learning** — Trained on a unified repository built from multiple public Alzheimer's MRI datasets.
- **Lightweight Transfer Learning** — EfficientNetB0, MobileNetV3, and ResNet backbones.
- **Ensemble Prediction** — Combines model outputs for improved robustness.
- **Explainable AI** — Grad-CAM heatmaps highlight the brain regions driving each prediction.
- **Confidence Estimation** — Every prediction includes a reliability score.
- **Clinical Decision Support Report** — Auto-generated summary with stage, explanation, and recommendation.
- **Progressive Disease Framework** — Predictions are presented within the Alzheimer's progression continuum.

---

## Classification Stages

Normal Classificaiton Stages: (First Few Tests)
| Stage | Label |
| :--- | :--- |
| 0 | Non Demented |
| 1 | Very Mild Demented |
| 2 | Mild Demented |
| 3 | Moderate Demented |

Tuned Classification Stages: 
| Stage | Label |
| :--- | :--- |
| 0 | Non Demented |
| 1 | EarlyDemented |
| 2 | Moderate Demented |

---

## System Pipeline
MRI Image
│
Image Preprocessing
│
Unified Multi-Dataset Repository
│
Train / Validation / Test Split
│
Ensemble Deep Learning Models
│
Final Classification
│
Confidence Score
│
Explainability (Grad-CAM)
│
Clinical Decision Support Report
---

## Tech Stack

- **Deep Learning:** TensorFlow / Keras
- **Backend:** Flask (Python)
- **Frontend:** HTML, CSS, JavaScript
- **Visualisation:** OpenCV, Matplotlib
- **Environment:** Google Colab (free-tier GPU)

---

## Project Status

This project is an academic research prototype developed as part of a final-year undergraduate project. It is **not intended for clinical or diagnostic use**.

---

## Disclaimer

NeuroInsight is a research and educational prototype. It does **not** replace professional medical evaluation. Always consult a qualified clinician for diagnosis and treatment decisions.

---

## Acknowledgements

- **OASIS** — Open Access Series of Imaging Studies
- **ADNI** — Alzheimer's Disease Neuroimaging Initiative
- **NACC** — National Alzheimer's Coordinating Center
- **AIBL** — Australian Imaging, Biomarker & Lifestyle Study of Ageing
- **Kaggle** — for hosting the public Alzheimer's MRI datasets used in this work

---

## License

This project is intended for academic and research purposes only.
