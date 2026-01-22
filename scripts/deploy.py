"""
Deployment Script for CBT Prediction API

Handles:
    1. Packaging Lambda function with dependencies
    2. Uploading model to S3
    3. Deploying CloudFormation stack
    4. Testing deployed endpoints

Usage:
    python scripts/deploy.py --action all --bucket my-cbt-bucket

Author: CBT Prediction Platform
Date: January 2026
"""

import os
import sys
import json
import shutil
import zipfile
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

import boto3
from botocore.exceptions import ClientError

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models" / "personal"  # Use fine-tuned model
DIST_DIR = PROJECT_ROOT / "dist"


def create_lambda_package() -> Path:
    """
    Create Lambda deployment package.
    
    Includes:
        - src/serving/ module
        - src/features/ module
        - Dependencies (numpy, pandas, scikit-learn, xgboost, etc.)
    
    Returns:
        Path to the created zip file
    """
    print("=" * 60)
    print("CREATING LAMBDA PACKAGE")
    print("=" * 60)
    
    package_dir = DIST_DIR / "lambda_package"
    
    # Clean previous builds
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True)
    
    # Install dependencies
    print("\n1. Installing dependencies...")
    requirements = [
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "joblib",
    ]
    
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        *requirements,
        "-t", str(package_dir),
        "--quiet"
    ], check=True)
    
    # Copy source code
    print("\n2. Copying source code...")
    
    # Copy serving module
    serving_dest = package_dir / "serving"
    shutil.copytree(SRC_DIR / "serving", serving_dest)
    
    # Copy features module
    features_dest = package_dir / "features"
    shutil.copytree(SRC_DIR / "features", features_dest)
    
    # Create handler at root level
    handler_content = '''
"""Lambda entry point."""
from serving.handler import lambda_handler

__all__ = ["lambda_handler"]
'''
    (package_dir / "handler.py").write_text(handler_content)
    
    # Create zip file
    print("\n3. Creating zip archive...")
    zip_path = DIST_DIR / f"lambda_function_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in package_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(package_dir)
                zf.write(file_path, arcname)
    
    package_size = zip_path.stat().st_size / (1024 * 1024)
    print(f"\n✓ Package created: {zip_path.name} ({package_size:.1f} MB)")
    
    # Clean up
    shutil.rmtree(package_dir)
    
    return zip_path


def upload_model_to_s3(bucket: str, prefix: str = "models/") -> None:
    """
    Upload trained model to S3.
    
    Uploads:
        - model.joblib
        - scaler.joblib
        - feature_names.json
        - metadata.json
    """
    print("=" * 60)
    print("UPLOADING MODEL TO S3")
    print("=" * 60)
    
    s3 = boto3.client("s3")
    
    model_files = [
        "model.joblib",
        "scaler.joblib",
        "feature_names.json",
        "metadata.json"
    ]
    
    for filename in model_files:
        local_path = MODELS_DIR / filename
        if not local_path.exists():
            print(f"  ⚠ Skipping {filename} (not found)")
            continue
        
        s3_key = f"{prefix}{filename}"
        
        print(f"  Uploading {filename}...")
        s3.upload_file(str(local_path), bucket, s3_key)
        print(f"  ✓ s3://{bucket}/{s3_key}")
    
    print("\n✓ Model uploaded successfully")


def upload_lambda_code(bucket: str, zip_path: Path) -> str:
    """Upload Lambda code to S3 and return the key."""
    s3 = boto3.client("s3")
    s3_key = f"lambda/{zip_path.name}"
    
    print(f"\nUploading Lambda code to s3://{bucket}/{s3_key}...")
    s3.upload_file(str(zip_path), bucket, s3_key)
    
    return s3_key


def deploy_cloudformation(
    bucket: str,
    fitbit_bucket: str,
    stack_name: str = "cbt-prediction-api"
) -> Dict:
    """
    Deploy CloudFormation stack.
    
    Returns:
        Stack outputs dict
    """
    print("=" * 60)
    print("DEPLOYING CLOUDFORMATION STACK")
    print("=" * 60)
    
    cf = boto3.client("cloudformation")
    
    template_path = PROJECT_ROOT / "infrastructure" / "cloudformation" / "main-stack.yaml"
    
    with open(template_path) as f:
        template_body = f.read()
    
    parameters = [
        {"ParameterKey": "ProjectName", "ParameterValue": "cbt-prediction"},
        {"ParameterKey": "FitbitBucket", "ParameterValue": fitbit_bucket},
        {"ParameterKey": "FitbitPrefix", "ParameterValue": ""},
    ]
    
    try:
        # Check if stack exists
        cf.describe_stacks(StackName=stack_name)
        action = "update"
    except ClientError:
        action = "create"
    
    print(f"\n{action.capitalize()}ing stack: {stack_name}")
    
    try:
        if action == "create":
            cf.create_stack(
                StackName=stack_name,
                TemplateBody=template_body,
                Parameters=parameters,
                Capabilities=["CAPABILITY_NAMED_IAM"]
            )
        else:
            cf.update_stack(
                StackName=stack_name,
                TemplateBody=template_body,
                Parameters=parameters,
                Capabilities=["CAPABILITY_NAMED_IAM"]
            )
        
        # Wait for completion
        print("Waiting for stack operation to complete...")
        waiter = cf.get_waiter(f"stack_{action}_complete")
        waiter.wait(StackName=stack_name)
        
        print(f"✓ Stack {action}d successfully!")
        
    except ClientError as e:
        if "No updates are to be performed" in str(e):
            print("No updates needed")
        else:
            raise
    
    # Get outputs
    response = cf.describe_stacks(StackName=stack_name)
    outputs = {}
    for output in response["Stacks"][0].get("Outputs", []):
        outputs[output["OutputKey"]] = output["OutputValue"]
    
    return outputs


def update_lambda_code(function_name: str, zip_path: Path) -> None:
    """Update Lambda function code."""
    print("=" * 60)
    print("UPDATING LAMBDA CODE")
    print("=" * 60)
    
    lambda_client = boto3.client("lambda")
    
    with open(zip_path, "rb") as f:
        zip_content = f.read()
    
    print(f"\nUpdating function: {function_name}")
    lambda_client.update_function_code(
        FunctionName=function_name,
        ZipFile=zip_content
    )
    
    print("✓ Lambda code updated")


def test_endpoints(api_endpoint: str) -> None:
    """Test deployed API endpoints."""
    import requests
    
    print("=" * 60)
    print("TESTING ENDPOINTS")
    print("=" * 60)
    
    # Health check
    print("\n1. Testing /health...")
    try:
        response = requests.get(f"{api_endpoint}/health", timeout=30)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Prediction test
    print("\n2. Testing /predict...")
    test_data = {
        "user_id": "test_user",
        "fitbit_data": [
            {"timestamp": "2026-01-13T10:00:00", "heart_rate": 72},
            {"timestamp": "2026-01-13T10:01:00", "heart_rate": 74},
            {"timestamp": "2026-01-13T10:02:00", "heart_rate": 73},
        ] * 10  # 30 samples
    }
    
    try:
        response = requests.post(
            f"{api_endpoint}/predict",
            json=test_data,
            timeout=30
        )
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Deploy CBT Prediction API to AWS"
    )
    
    parser.add_argument(
        "--action",
        choices=["package", "upload-model", "deploy", "update-code", "test", "all"],
        default="all",
        help="Deployment action"
    )
    parser.add_argument(
        "--bucket",
        required=True,
        help="S3 bucket for model and predictions"
    )
    parser.add_argument(
        "--fitbit-bucket",
        help="S3 bucket with Fitbit data (defaults to --bucket)"
    )
    parser.add_argument(
        "--stack-name",
        default="cbt-prediction-api",
        help="CloudFormation stack name"
    )
    parser.add_argument(
        "--api-endpoint",
        help="API endpoint for testing (auto-detected if deploying)"
    )
    
    args = parser.parse_args()
    
    if args.fitbit_bucket is None:
        args.fitbit_bucket = args.bucket
    
    # Ensure dist directory exists
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    
    outputs = {}
    zip_path = None
    
    if args.action in ["package", "all"]:
        zip_path = create_lambda_package()
    
    if args.action in ["upload-model", "all"]:
        upload_model_to_s3(args.bucket)
    
    if args.action in ["deploy", "all"]:
        outputs = deploy_cloudformation(
            args.bucket,
            args.fitbit_bucket,
            args.stack_name
        )
        print("\nStack Outputs:")
        for key, value in outputs.items():
            print(f"  {key}: {value}")
    
    if args.action in ["update-code", "all"]:
        if zip_path is None:
            zip_path = create_lambda_package()
        
        function_name = outputs.get("LambdaName", f"{args.stack_name.replace('-stack', '')}-api")
        update_lambda_code(function_name, zip_path)
    
    if args.action in ["test", "all"]:
        endpoint = args.api_endpoint or outputs.get("ApiEndpoint")
        if endpoint:
            test_endpoints(endpoint)
        else:
            print("\n⚠ No API endpoint available for testing")
    
    print("\n" + "=" * 60)
    print("DEPLOYMENT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()