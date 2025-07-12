# TumorTracer: MLOps-Ready Lung Cancer Classifier
TumorTracer is an MLOps-first project that classifies lung tumor types using CT scan images. It integrates the full machine learning lifecycle—from data ingestion to model selection, versioning, and deployment—following modular and production-ready practices.

Built using:  
🛠️ **TensorFlow/Keras**, 🧪 **MLflow**, 📦 **DVC**, 🐳 **Docker**, 🔁 **GitHub Actions**, 🧠 **Flask API**

## 🧬 Problem Statement

The goal is to classify a given chest CT scan into one of four categories:

- Adenocarcinoma
- Large Cell Carcinoma
- Squamous Cell Carcinoma
- Normal

## 🌐 Project Architecture

```text
TumorTracer
├── src/cnnClassifier                   # All pipeline components
│   ├── components                      # Model building, training, predictions
│   ├── config                          # YAML config parsing
│   ├── constants                       # Static config values
│   ├── entity                          # Config data classes
│   ├── pipeline                        # Stage-wise scripts for DVC
│   └── utils                           # Helper functions
├── artifacts                           # All intermediate + final artifacts
│   ├── base_model                      # Base + updated transfer models
│   ├── data_ingestion                  # Raw + preprocessed datasets
│   ├── model_training                  # Checkpoints, logs, selected model
│   └── user_inputs                     # Uploaded images via API
├── model                               # Final selected model + class index map
├── research                            # Jupyter notebooks for experimentation
├── config/config.yaml                  # Paths and directory mapping
├── params/params.yaml                  # Hyperparameters and augmentation
├── app.py                              # Flask prediction API
├── dvc.yaml, dvc.lock                  # DVC pipeline config
├── Dockerfile                          # Production Docker build
├── requirements.txt                    # Dependencies
└── .github/workflows/workflow.yml      # CI/CD pipeline

