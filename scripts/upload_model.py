"""Upload model to S3."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import boto3
from config import MODEL_BUCKET, MODEL_PREFIX, MODELS_DIR, require_env


def upload_model():
    bucket = require_env("MODEL_BUCKET")
    s3 = boto3.client("s3")
    
    files = ["model.joblib", "scaler.joblib", "features.json", "model_metadata.json"]
    
    print(f"Uploading to s3://{bucket}/{MODEL_PREFIX}")
    
    for f in files:
        path = MODELS_DIR / f
        if path.exists():
            s3.upload_file(str(path), bucket, f"{MODEL_PREFIX}{f}")
            print(f"  ✓ {f}")
        else:
            print(f"  ⚠ {f} not found (skipped)")


if __name__ == "__main__":
    upload_model()