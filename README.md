# Core Body Temperature (CBT) Prediction Pipeline

This repository is designed to predict core body temperature (CBT) in real time from physiological data (via Fitbit) and environmental data, using a trained machine learning model. It includes the full pipeline:

data ingestion → preprocessing → feature transformation → model serving (AWS Lambda)

This repository is designed to be reproducible, usable for multiple users during the ingestion stage, deployment-ready on AWS Lambda and S3, and friendly for contributors to test and validate the scripts. In addition, the transformation step is designed to stay consistent so the same feature logic is used in both local runs and deployment.

## 1. What’s included and what it does

This repository implements a **core body temperature (CBT) prediction platform** built around a clear, end-to-end machine learning workflow.

At a high level, it includes:

* **Data preprocessing** to clean and standardize raw physiological (Fitbit) and environmental data
* **Feature engineering** to transform time-series signals into the exact feature set used by the model
* **Training pipelines** for:

  * external research data (PROSPIE)
  * study- or user-collected data (fine-tuning)
* **Model serving (inference)** code that runs in AWS Lambda and produces CBT predictions on demand
* **Tests** that verify feature correctness, data validity, and protection against data leakage

The main purpose of the repo is to ensure that **the same feature logic is used consistently across training and prediction**, and that CBT predictions can be reliably generated for multiple users in a production setting.

---

## 2. Getting started

1. **Clone the repository**

   ```bash
   git clone <repo-url>
   cd cbt_pred_platform
   ```

2. **Set up a virtual environment and install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Prepare data**

   * Place raw CSV files into the expected raw input folders
   * Run the preprocessing step to normalize timestamps, formats, and structure:

     ```bash
     python preprocessing/preprocessing.py
     ```

4. **Build model-ready data**

   * Convert preprocessed files into training features aligned to CBT measurements:

     ```bash
     python -m src.training.prepare_data
     ```

At this point, the pipeline is ready for training, evaluation, or prediction experiments.

---

## 3. How to test

To verify that everything is working correctly, run the test suite:

```bash
pytest
```

The tests confirm that:

* The correct number of features is produced
* Feature names match the expected specification
* All required signals are present
* Features are computed using only past data (no leakage)
* Unit conversions and time handling are correct

If all tests pass, the pipeline is producing valid, model-compatible inputs and is safe to use for training or inference.

## Folder structure for cbt_pred_platform

```text
cbt_pred_platform/
├── data/
│   └── raw/                      # PROSPIE-derived processed training data (external experiment)
│
├── fitbit_to_aws_s3/             # Fitbit multi-user ingestion pipeline
│   ├── main.py                   # FastAPI OAuth2 server for Fitbit user onboarding
│   ├── fitbit_multi_to_s3.py     # Multi-user Fitbit data fetch → upload to S3
│   ├── token_manager.py          # Token storage and refresh handling
│   ├── requirements.txt
│   ├── .env.example              # Example environment variable configuration
│   └── README.md                 # Fitbit ingestion-specific documentation
│
├── preprocessing/                # Stable normalization layer (DO NOT casually change)
│   └── preprocessing.py          # Normalizes raw input data for downstream transformation
│
├── scripts/                      # AWS automation and deployment utilities
│   ├── setup_aws.py              # Initial AWS infrastructure setup
│   ├── deploy.py                 # Lambda packaging and deployment
│   └── upload_model.py           # Upload trained model artifacts to S3
│
├── src/                          # Core package: features, serving, training
│   ├── features/                 # Feature engineering and transformation logic
│   │   ├── __init__.py
│   │   └── transformations.py    # Converts preprocessed data into model-ready features
│   │
│   ├── serving/                  # Lambda serving components (API + data loaders)
│   │   ├── __init__.py
│   │   ├── fitbit_loader.py      # Loads user Fitbit data from S3 for inference
│   │   ├── handler.py            # AWS Lambda entry point and API routing
│   │   └── predictor.py          # Model loading, caching, and prediction logic
│   │
│   └── training/                 # Model training and evaluation utilities
│       ├── __init__.py
│       ├── evaluate.py           # Model performance evaluation
│       ├── finetune_model.py     # Optional fine-tuning logic
│       ├── prepare_data.py       # Training data assembly and alignment
│       ├── prepare_external_data.py # External dataset preparation (e.g., PROSPIE)
│       └── train_model.py        # Model training entry point
│
├── tests/                        # Feature transformation and loader validation
│   └── test_transformations.py   # Tests for feature correctness and stability
│
├── requirements.txt              # Top-level Python dependencies
└── README.md                     # Repository overview and onboarding guide


```

## 4. Status 🚧

The inference pipeline for testing CBT predictions is currently under development.



