# TumorTracer: MLOps-Ready Lung Cancer Classifier
TumorTracer is an MLOps-first project that classifies lung tumor types using CT scan images. It integrates the full machine learning lifecycle—from data ingestion to model selection, versioning, and deployment—following modular and production-ready practices.


## ⚠️ Disclaimer

This project is intended strictly for **educational and research purposes**.

While the model demonstrates the use of deep learning for medical image classification (specifically CT scans of lung tumors), it is **not validated for clinical use** and should **not be used for real-world diagnosis, treatment, or medical decision-making**.

The predictions made by this application are **not reviewed by medical professionals**, and **no liability is accepted** for misuse of this software.

For any medical concerns, always consult a **qualified healthcare provider**.


## 🌐 Built With
🛠️ **TensorFlow/Keras**, 🧪 **MLflow**, 📦 **DVC**, 🐳 **Docker**, 🔁 **GitHub Actions**, 🧠 **Flask API**, ☁️ **AWS**


## 🧬 Problem Statement

The goal is to classify a given chest CT scan into one of four categories:

- Adenocarcinoma
- Large Cell Carcinoma
- Squamous Cell Carcinoma
- Normal (i.e., the CT scan could be classified into the other three categories)

Using the base VGG16 model only results in a validation accuracy score of `0.2448` and a test accuracy score of `0.2381`.


## 📌 Project Highlights

- **Modular ML Pipeline**: The entire machine learning workflow is broken down into modular stages using Python classes and DVC pipelines:
  - Data ingestion from Kaggle
  - Base model preparation (VGG16)
  - Training with callbacks and checkpoints
  - Model selection based on highest validation accuracy
  - Prediction serving via Flask

- **Version Control with DVC**: Every stage of the ML pipeline—data, models, and configurations—is tracked using DVC, ensuring reproducibility and easy rollbacks.

- **Experiment Tracking with MLflow**: All training runs, hyperparameters, metrics, and model artifacts are logged with MLflow. The best model is registered automatically.

- **Smart Checkpointing & Selection**: 
  - Each training epoch saves a `.keras` checkpoint with accuracy and val_accuracy in the filename.
  - A regex-based selector parses all checkpoints to find the one with the highest validation accuracy, and deploys it automatically.

- **Model Selection Automation**: Automatically parses training checkpoint filenames using regex to identify and copy the best-performing model to production.

- **Pretrained Model (VGG16)**: Leverages `imagenet` weights for initialization, then fine-tunes for lung tumor classification across 4 categories.

- **Image-Based Predictions via REST API**: Users can send Base64-encoded images to the Flask API to receive real-time tumor classification results with confidence scores.

- **Organized Project Structure**: Follows industry-grade project structuring with clearly separated modules (`components`, `entity`, `pipeline`, `utils`, etc.).

- **Dockerized**: Easily deployable anywhere using Docker. Includes GitHub Actions workflow for CI/CD and EC2-based Continuous Deployment.

- **Robust Logging**: Centralized logging system that tracks errors, training progress, and pipeline execution for both test and running environments.


## 🧱 Project Architecture

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


## 🧬 Dataset Description

The model is trained on the **Chest CTScan Images** dataset available on [Kaggle](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images). This dataset consists of CT scan images categorized into four classes related to lung health

### Structure

The dataset is divided into three standard subsets:

    chest-ctscan-images/
    ├── test/
    ├── train/
    └── valid/

Each subset contains images organized in subfolders based on their labels. For example:

    train/
    ├── adenocarcinoma/
    ├── large_cell_carcinoma/
    ├── squamous_cell_carcinoma/
    └── normal/

### Characteristics

- **Format**: JPEG/PNG medical images
- **Type**: Labeled classification dataset (supervised learning)
- **Use case**: Multi-class classification of lung tumor types from CT scan images
- **Class balance**: The dataset has moderate imbalance across classes, addressed via data augmentation 

### Class Name Normalization

Some folders in the raw dataset contain verbose or complex names. These are simplified and standardized using a YAML-based mapping file at: `config/class_name_map.yaml`

## ⚙️ MLOps Pipeline (via DVC)

This project implements a reproducible and modular machine learning pipeline using **DVC (Data Version Control)**. DVC ensures that data, models, and intermediate artifacts are version-controlled and that the pipeline stages are tracked effectively.

### Stages in the DVC Pipeline

1. **Data Ingestion**
   - Downloads the dataset from Kaggle.
   - Extracts and organizes the dataset into train, validation, and test folders.
   - Defined in: `stage_01_data_ingestion.py`

2. **Base Model Preparation**
   - Loads a pre-trained VGG16 model (without top layers).
   - Freezes initial layers and appends custom classification layers.
   - Defined in: `stage_02_base_model.py`

3. **Model Training**
   - Trains the model using the prepared dataset.
   - Applies data augmentation, if specified.
   - Uses MLflow for experiment tracking, if specified.
   - Saves checkpoints to `artifacts/model_training/Checkpoint_*`, if specified.
   - Defined in: `stage_03_model_training.py`

4. **Model Selection**
   - Scans through all model checkpoints.
   - Selects the best model based on highest validation accuracy.
   - Copies selected model and class indices to the `model/` directory.
   - Defined in: `stage_04_model_selection.py`

### Pipeline Management

- The pipeline is defined in `dvc.yaml` and tracked using `dvc.lock`.
- Configuration is split into:
  - `config.yaml` – file paths and structure
  - `params.yaml` – tunable hyperparameters
- Reproducibility is ensured by running: `dvc repro`


## 🧪 MLflow Integration

This project uses **MLflow** to track experiments, log hyperparameters, monitor metrics, and register models for deployment.

### Key Capabilities Used

- **Autologging with TensorFlow/Keras**  
  MLflow automatically captures:
  - Training & validation metrics per epoch (accuracy, loss, etc.)
  - Optimizer and learning rate
  - Input shapes and model summary
  - Artifacts like trained models

- **MLflow Callbacks**  
  Custom callbacks are integrated during model training (`stage_03_model_training.py`) to ensure that each run is logged in a consistent and structured way.

- **Model Registry**  
  Models can be pushed to an MLflow-compatible model registry. This enables versioned model management, experimentation tracking, and simplified deployment.

- **Parameter Tracking**  
  Hyperparameters (e.g., batch size, epochs, optimizer, augmentation settings) are defined in `params/params.yaml` and logged for every training session.

### Example Logged Metrics

| Epoch | Accuracy | Validation Accuracy | Loss    | Validation Loss |
|-------|----------|---------------------|---------|-----------------|
| 1     | 0.4617   | 0.5278              | 10.2913 | 4.2525          |
| 2     | 0.6509   | 0.6111              | 3.3595  | 3.3058          |
| 3     | 0.7341   | 0.5555              | 2.0722  | 6.2245          |
| 4     | 0.7406   | 0.6389              | 2.9830  | 3.1352          |
| 5     | 0.8124   | 0.7917              | 1.3135  | 2.7899          |

## 🧠 Model Selection Logic

After each training run, multiple model checkpoints are saved with filenames that embed their training and validation metrics. The model selection logic is designed to automatically identify and promote the best model based on **highest validation accuracy (vacc)**.

### How It Works

- Each model checkpoint is named using a consistent pattern:  
  `model_e{epoch}_acc{accuracy}_vacc{validation_accuracy}.keras`
- A **regex pattern** defined in `params/params.yaml` is used to extract the `vacc` value from each model filename.
- A script scans all checkpoint subfolders (e.g., `Checkpoint_20250712_1045/`) under `artifacts/model_training/`.
- The model with the **highest `vacc`** is selected.
- Along with the best `.keras` model, the corresponding `class_indices.json` file from the same checkpoint is also selected.
- Both selected files are copied into the `model/` directory for prediction and deployment.


## 📦 API Inference (Flask)

The project includes a lightweight RESTful API built with **Flask** to allow users to submit medical images (as base64 strings) and receive real-time tumor type predictions.

### Inference Flow

1. Incoming base64 image is decoded and saved temporarily.
2. The saved image is passed to the best model in `model/trained_model.keras`.
3. The `PredictionPipeline` class handles image preprocessing and prediction.
4. Confidence scores are computed and returned.
5. The temporary image file is deleted after prediction.

### CORS Enabled

CORS support is included via `Flask-CORS`, allowing cross-origin requests from external clients or front-end apps.

### Input Format

- The model expects images sized to `(224, 224, 3)` (as per VGG16 architecture).
- Preprocessing includes scaling pixel values using a `NORMALIZATION` factor (defined in `params.yaml`, and as used during training).

### Retraining on Demand

- Hitting the `POST /train` endpoint will re-trigger the full DVC pipeline.
- The model is retrained, and the new best checkpoint is selected and deployed — fully automating 

### API Use Cases

- Web-based user interface
- Integration with hospital systems
- Rapid testing for new image uploads
- Auto-retraining interface using `/train`


## ☁️ AWS Infrastructure

- **Amazon ECR**: Hosts container images.
- **Amazon EC2 (Self-Hosted Runner)**: Pulls latest model image and serves the Flask API.
- **GitHub Actions**: Automates end-to-end lifecycle.

## 🛠️ Local Setup
Follow these steps to set up and run the project locally:

**1. Clone the Repository**

```bash
git clone https://github.com/Vish501/TumorTracer.git
```
```bash
cd TumorTracer
```

**2. Setup Virtual Environment (Optional)**

It's recommended to use a virtual environment to manage dependencies. You can create and activate one using:

```bash
conda create -p venv python=3.12.1 -y
```
```bash
conda activate venv/
```

**3. Install Dependencies**

Install the required packages:

```bash
pip install -r requirements.txt
```

**4. Configure Environment Variables**

1. Create a .env file in your main directory `touch .env`.
2. Go to the .env file `code .env`.
3. Update the file with your Kaggle JSON credentials as a string, with the variable name `KAGGLE_JSON`.
4. Update the file with your Working Directory path, with the variable name `WORKING_DIRECTORY`.

**5. Run DVC Pipeline**

To download data, prepare model, and train from scratch:

```bash
dvc repro
```

**6. Run the Flask App**

```bash
python app.py
```

App will be available at: [http://localhost:8080](http://localhost:8080)

**7. Access the API**

You can now:

- Predict: Send a POST request to `/predict` with an image in Base64 format
- Retrain: Send a POST request to `/train` to retrigger the pipeline


## 📈 Future Enhancements

This project provides a solid foundation for end-to-end medical image classification, but there are several opportunities to enhance its capabilities further:

**1. Model Improvements**
- **Advanced Architectures**: Integrate architectures like ResNet, EfficientNet, or Vision Transformers (ViT).
- **Ensemble Learning**: Combine multiple models to improve generalization.
- **AutoML**: Use tools like Keras Tuner, Optuna, or Google AutoML for hyperparameter optimization.

**2. Explainability**
- **Model Interpretability**: Add Grad-CAM or LIME to visualize which regions of the CT scan influence predictions.
- **Prediction Logs**: Store past predictions with timestamps and confidence scores for auditability.

**3. UI Enhancements**
- **Upload Interface**: Enhance the frontend to support drag-and-drop image upload.
- **Display Predictions**: Render prediction results on the interface instead of just returning JSON.

**4. Deployment Improvements**
- **Monitoring**: Integrate Prometheus and Grafana for real-time performance monitoring.
- **HTTPS & Auth**: Secure API endpoints and UI with HTTPS and authentication (OAuth or API key).

**5. CI/CD Expansion**
- **Model Evaluation Gate**: Only deploy if validation accuracy improves over the last registered model.
- **Staging Environment**: Add a staging EC2 instance for QA before production.

`If you’re interested in contributing to any of these features, feel free to fork the repository and raise a pull request.`

## 🙋‍♂️ Author

This project was developed and maintained by **Vish501**, a data science enthusiast with a background in finance and a strong focus on MLOps, deep learning, and end-to-end machine learning systems.

Feel free to contribute to this project by submitting issues or pull requests. For any questions or suggestions, please contact Vish501.