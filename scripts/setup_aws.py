"""
AWS Infrastructure Setup for CBT Prediction Platform
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError, WaiterError
except ImportError:
    print("ERROR: boto3 not installed. Run: pip install boto3")
    sys.exit(1)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models" / "study_model_v1"
INFRASTRUCTURE_DIR = PROJECT_ROOT / "infrastructure" / "cloudformation"
DIST_DIR = PROJECT_ROOT / "dist"


class AWSSetup:
    """Handles AWS infrastructure setup for CBT Prediction Platform."""
    
    def __init__(self, region: str = "us-east-2", project_name: str = "cbt-prediction"):
        self.region = region
        self.project_name = project_name
        self.stack_name = f"{project_name}-stack"
        
        self.s3 = boto3.client("s3", region_name=region)
        self.cf = boto3.client("cloudformation", region_name=region)
        self.lambda_client = boto3.client("lambda", region_name=region)
        self.sts = boto3.client("sts", region_name=region)
        
        self.account_id = self.sts.get_caller_identity()["Account"]
        self.bucket_name = f"{project_name}-{self.account_id}"
        
        print(f"AWS Setup initialized:")
        print(f"  Region:     {self.region}")
        print(f"  Account:    {self.account_id}")
        print(f"  Bucket:     {self.bucket_name}")
        print(f"  Stack:      {self.stack_name}")
        print()
    
    def create_bucket(self) -> bool:
        """Create S3 bucket for models with retry logic."""
        print("=" * 60)
        print("CREATING S3 BUCKET")
        print("=" * 60)
        
        # Check if bucket already exists and we have access
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
            print(f"Bucket already exists: {self.bucket_name}")
            return True
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                pass  # Bucket doesn't exist, we'll create it
            elif error_code == "403":
                print(f"ERROR: Bucket {self.bucket_name} exists but you don't have access")
                return False
            elif error_code == "400":
                pass  # Bad request, try to create
            else:
                print(f"Head bucket error: {error_code}")
        
        # Retry logic for bucket creation
        max_retries = 5
        for attempt in range(max_retries):
            try:
                print(f"Creating bucket (attempt {attempt + 1}/{max_retries})...")
                
                # us-east-1 doesn't need LocationConstraint, all others do
                if self.region == "us-east-1":
                    self.s3.create_bucket(Bucket=self.bucket_name)
                else:
                    self.s3.create_bucket(
                        Bucket=self.bucket_name,
                        CreateBucketConfiguration={"LocationConstraint": self.region}
                    )
                
                print(f"Created bucket: {self.bucket_name}")
                
                # Wait for bucket to be available
                print("Waiting for bucket to be ready...")
                waiter = self.s3.get_waiter("bucket_exists")
                waiter.wait(Bucket=self.bucket_name)
                
                print("Bucket is ready!")
                return True
                
            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                error_msg = str(e)
                
                if "OperationAborted" in error_msg or "conflicting" in error_msg.lower():
                    # Bucket operation in progress, wait and retry
                    wait_time = 10 * (attempt + 1)
                    print(f"  Bucket operation in progress, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                    
                elif "BucketAlreadyOwnedByYou" in error_msg:
                    print(f"Bucket already exists and owned by you: {self.bucket_name}")
                    return True
                    
                elif "BucketAlreadyExists" in error_msg:
                    print(f"ERROR: Bucket name already taken globally: {self.bucket_name}")
                    return False
                    
                else:
                    print(f"ERROR creating bucket: {e}")
                    if attempt < max_retries - 1:
                        print(f"  Retrying in 10s...")
                        time.sleep(10)
                    else:
                        return False
        
        print("ERROR: Failed to create bucket after all retries")
        return False
    
    def upload_model(self) -> bool:
        """Upload trained model to S3."""
        print("=" * 60)
        print("UPLOADING MODEL TO S3")
        print("=" * 60)
        
        if not MODELS_DIR.exists():
            print(f"WARNING: Model directory not found: {MODELS_DIR}")
            print("Skipping model upload - you can upload later")
            return True  # Don't fail the whole setup
        
        files = ["model.joblib", "scaler.joblib", "feature_names.json", "metadata.json"]
        uploaded = 0
        
        for filename in files:
            local_path = MODELS_DIR / filename
            if not local_path.exists():
                print(f"  Skipping {filename} (not found)")
                continue
            
            try:
                print(f"  Uploading {filename}...")
                self.s3.upload_file(str(local_path), self.bucket_name, f"models/{filename}")
                print(f"    -> s3://{self.bucket_name}/models/{filename}")
                uploaded += 1
            except ClientError as e:
                print(f"  ERROR uploading {filename}: {e}")
        
        print(f"\nUploaded {uploaded} files")
        return True
    
    def _delete_stack_if_failed(self) -> bool:
        """Delete stack if it's in a failed state."""
        try:
            response = self.cf.describe_stacks(StackName=self.stack_name)
            status = response["Stacks"][0]["StackStatus"]
            
            print(f"Current stack status: {status}")
            
            failed_states = [
                "ROLLBACK_COMPLETE", "ROLLBACK_FAILED", 
                "CREATE_FAILED", "DELETE_FAILED",
                "UPDATE_ROLLBACK_COMPLETE", "UPDATE_ROLLBACK_FAILED"
            ]
            
            in_progress_states = [
                "CREATE_IN_PROGRESS", "DELETE_IN_PROGRESS",
                "UPDATE_IN_PROGRESS", "ROLLBACK_IN_PROGRESS",
                "UPDATE_ROLLBACK_IN_PROGRESS"
            ]
            
            if status in in_progress_states:
                print(f"Stack operation in progress. Waiting...")
                # Wait for current operation to complete
                for i in range(60):
                    time.sleep(10)
                    try:
                        response = self.cf.describe_stacks(StackName=self.stack_name)
                        status = response["Stacks"][0]["StackStatus"]
                        print(f"  Status: {status}")
                        if status not in in_progress_states:
                            break
                    except ClientError:
                        print("  Stack no longer exists")
                        return True
            
            if status in failed_states:
                print(f"Stack in failed state ({status}), deleting...")
                self.cf.delete_stack(StackName=self.stack_name)
                
                # Wait for deletion
                print("Waiting for stack deletion...")
                for i in range(120):
                    time.sleep(5)
                    try:
                        response = self.cf.describe_stacks(StackName=self.stack_name)
                        status = response["Stacks"][0]["StackStatus"]
                        if i % 6 == 0:
                            print(f"  Status: {status}")
                    except ClientError as e:
                        if "does not exist" in str(e):
                            print("Stack deleted successfully.")
                            return True
                
                print("WARNING: Stack deletion timed out")
                return False
            
            return True  # Stack exists and is healthy
            
        except ClientError as e:
            if "does not exist" in str(e):
                print("No existing stack found.")
                return True  # No stack exists, good to create
            raise
    
    def deploy_stack(self) -> dict:
        """Deploy CloudFormation stack."""
        print("=" * 60)
        print("DEPLOYING CLOUDFORMATION STACK")
        print("=" * 60)
        
        template_path = INFRASTRUCTURE_DIR / "main-stack.yaml"
        if not template_path.exists():
            print(f"ERROR: Template not found: {template_path}")
            return {}
        
        # Handle any existing failed stacks
        if not self._delete_stack_if_failed():
            print("ERROR: Could not clean up existing stack")
            return {}
        
        # Small delay after cleanup
        time.sleep(5)
        
        with open(template_path, encoding="utf-8") as f:
            template_body = f.read()
        
        # Check if stack exists
        stack_exists = False
        try:
            response = self.cf.describe_stacks(StackName=self.stack_name)
            status = response["Stacks"][0]["StackStatus"]
            if status.endswith("_COMPLETE") and "ROLLBACK" not in status:
                stack_exists = True
        except ClientError:
            pass
        
        action = "update" if stack_exists else "create"
        print(f"\n{action.title()}ing stack: {self.stack_name}")
        
        try:
            params = {
                "StackName": self.stack_name,
                "TemplateBody": template_body,
                "Parameters": [
                    {"ParameterKey": "ProjectName", "ParameterValue": self.project_name}
                ],
                "Capabilities": ["CAPABILITY_NAMED_IAM"]
            }
            
            if action == "create":
                params["OnFailure"] = "DELETE"
                self.cf.create_stack(**params)
            else:
                self.cf.update_stack(**params)
            
            # Wait with progress updates
            print("Waiting for stack operation... (2-5 minutes)")
            for i in range(90):
                time.sleep(10)
                try:
                    response = self.cf.describe_stacks(StackName=self.stack_name)
                    status = response["Stacks"][0]["StackStatus"]
                    
                    if i % 3 == 0:
                        print(f"  Status: {status} ({i*10}s)")
                    
                    if status in ["CREATE_COMPLETE", "UPDATE_COMPLETE"]:
                        print(f"\nStack {action}d successfully!")
                        break
                    elif "ROLLBACK" in status or "FAILED" in status:
                        print(f"\nERROR: Stack {action} failed: {status}")
                        self._print_stack_errors()
                        return {}
                        
                except ClientError as e:
                    if "does not exist" in str(e):
                        print("\nERROR: Stack was deleted (creation failed)")
                        return {}
            
        except ClientError as e:
            error_msg = str(e)
            if "No updates" in error_msg:
                print("No updates needed - stack is current")
            elif "already exists" in error_msg:
                print("Stack already exists, checking outputs...")
            else:
                print(f"ERROR: {e}")
                self._print_stack_errors()
                return {}
        
        # Get outputs
        try:
            response = self.cf.describe_stacks(StackName=self.stack_name)
            outputs = {o["OutputKey"]: o["OutputValue"] 
                      for o in response["Stacks"][0].get("Outputs", [])}
            
            print("\nStack Outputs:")
            for k, v in outputs.items():
                print(f"  {k}: {v}")
            
            return outputs
        except ClientError as e:
            print(f"Warning: Could not get outputs: {e}")
            return {}
    
    def _print_stack_errors(self):
        """Print recent stack errors."""
        try:
            events = self.cf.describe_stack_events(StackName=self.stack_name)
            print("\nRecent stack events:")
            count = 0
            for event in events["StackEvents"]:
                status = event.get("ResourceStatus", "")
                if "FAILED" in status or "ROLLBACK" in status:
                    reason = event.get("ResourceStatusReason", "No reason given")
                    resource = event.get("LogicalResourceId", "Unknown")
                    print(f"  [{status}] {resource}: {reason}")
                    count += 1
                    if count >= 10:
                        break
        except Exception as e:
            print(f"Could not get stack events: {e}")
    
    def create_lambda_package(self) -> Path:
        """Create minimal Lambda package."""
        print("=" * 60)
        print("CREATING LAMBDA PACKAGE")
        print("=" * 60)
        
        DIST_DIR.mkdir(parents=True, exist_ok=True)
        package_dir = DIST_DIR / "lambda_package"
        
        if package_dir.exists():
            shutil.rmtree(package_dir)
        package_dir.mkdir(parents=True)
        
        # Install minimal dependencies
        print("\n1. Installing dependencies...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "joblib==1.3.2", "scikit-learn==1.4.0", "numpy==1.26.4",
             "-t", str(package_dir), "--quiet", "--no-cache-dir"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"  Warning: {result.stderr[:200]}")
        
        # Clean up to reduce size
        print("2. Cleaning up unnecessary files...")
        patterns_to_remove = ["*.dist-info", "__pycache__", "tests", "test", "*.pyc", "*.pyo"]
        for pattern in patterns_to_remove:
            for p in package_dir.rglob(pattern):
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                elif p.is_file():
                    p.unlink(missing_ok=True)
        
        # Remove large unnecessary directories
        large_dirs = ["numpy/tests", "numpy/doc", "sklearn/datasets", "sklearn/tests"]
        for d in large_dirs:
            dir_path = package_dir / d
            if dir_path.exists():
                shutil.rmtree(dir_path, ignore_errors=True)
        
        # Create handler
        print("3. Creating Lambda handler...")
        handler_code = '''import json
import os
import traceback

def lambda_handler(event, context):
    """Main Lambda handler for CBT Prediction API."""
    try:
        # Extract path and method
        path = event.get('rawPath', event.get('path', '/'))
        method = event.get('requestContext', {}).get('http', {}).get('method', 'GET')
        
        # Health check endpoint
        if '/health' in path:
            return make_response(200, {
                'status': 'healthy',
                'version': '1.0.0',
                'model_bucket': os.environ.get('MODEL_BUCKET', 'not-configured'),
                'region': os.environ.get('AWS_REGION', 'unknown')
            })
        
        # Prediction endpoint
        if '/predict' in path and method == 'POST':
            # Parse request body
            body = event.get('body', '{}')
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except json.JSONDecodeError:
                    return make_response(400, {'error': 'Invalid JSON in request body'})
            
            # For now, return a placeholder response
            # TODO: Load model from S3 and make actual predictions
            return make_response(200, {
                'status': 'success',
                'message': 'Prediction endpoint ready',
                'received_fields': list(body.keys()) if isinstance(body, dict) else [],
                'note': 'Actual model inference will be implemented'
            })
        
        # Batch prediction endpoint
        if '/batch' in path and method == 'POST':
            return make_response(200, {
                'status': 'success',
                'message': 'Batch endpoint ready'
            })
        
        # Not found
        return make_response(404, {
            'error': 'Not found',
            'path': path,
            'method': method,
            'available_endpoints': ['GET /health', 'POST /predict', 'POST /batch']
        })
        
    except Exception as e:
        return make_response(500, {
            'error': 'Internal server error',
            'message': str(e),
            'traceback': traceback.format_exc()
        })

def make_response(status_code, body):
    """Create a properly formatted API Gateway response."""
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        },
        'body': json.dumps(body)
    }
'''
        (package_dir / "handler.py").write_text(handler_code)
        
        # Create zip
        print("4. Creating zip archive...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_path = DIST_DIR / f"lambda_{timestamp}.zip"
        
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for f in package_dir.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(package_dir))
        
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"\nPackage created: {zip_path.name}")
        print(f"Size: {size_mb:.1f} MB")
        
        # Cleanup
        shutil.rmtree(package_dir)
        
        return zip_path
    
    def update_lambda(self, zip_path: Path = None) -> bool:
        """Update Lambda function code."""
        print("=" * 60)
        print("UPDATING LAMBDA CODE")
        print("=" * 60)
        
        function_name = f"{self.project_name}-api"
        
        # Check if Lambda exists
        try:
            self.lambda_client.get_function(FunctionName=function_name)
            print(f"Found Lambda function: {function_name}")
        except ClientError as e:
            if "ResourceNotFoundException" in str(e):
                print(f"ERROR: Lambda '{function_name}' not found.")
                print("Make sure the CloudFormation stack deployed successfully.")
                print("Run: python scripts/setup_aws.py --action deploy-stack")
                return False
            raise
        
        # Create package if not provided
        if zip_path is None or not zip_path.exists():
            zip_path = self.create_lambda_package()
        
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        
        try:
            if size_mb > 50:
                # Upload via S3 for large packages
                print(f"Package {size_mb:.1f}MB - uploading via S3...")
                s3_key = f"lambda-code/{zip_path.name}"
                self.s3.upload_file(str(zip_path), self.bucket_name, s3_key)
                print(f"  Uploaded to s3://{self.bucket_name}/{s3_key}")
                
                self.lambda_client.update_function_code(
                    FunctionName=function_name,
                    S3Bucket=self.bucket_name,
                    S3Key=s3_key
                )
            else:
                # Direct upload for smaller packages
                print(f"Package {size_mb:.1f}MB - uploading directly...")
                with open(zip_path, "rb") as f:
                    self.lambda_client.update_function_code(
                        FunctionName=function_name,
                        ZipFile=f.read()
                    )
            
            # Wait for update to complete
            print("Waiting for Lambda update to complete...")
            for i in range(30):
                time.sleep(3)
                resp = self.lambda_client.get_function(FunctionName=function_name)
                state = resp["Configuration"].get("State", "Unknown")
                update_status = resp["Configuration"].get("LastUpdateStatus", "Unknown")
                
                if state == "Active" and update_status == "Successful":
                    print(f"\nLambda updated successfully!")
                    print(f"  Code size: {resp['Configuration']['CodeSize'] / 1024 / 1024:.1f} MB")
                    return True
                elif update_status == "Failed":
                    reason = resp["Configuration"].get("LastUpdateStatusReason", "Unknown")
                    print(f"\nERROR: Lambda update failed: {reason}")
                    return False
                
                if i % 3 == 0:
                    print(f"  State: {state}, Update: {update_status}")
            
            print("Lambda update completed")
            return True
            
        except ClientError as e:
            print(f"ERROR updating Lambda: {e}")
            return False
    
    def test_api(self, endpoint: str = None) -> bool:
        """Test the API endpoints."""
        print("=" * 60)
        print("TESTING API")
        print("=" * 60)
        
        # Get endpoint from stack if not provided
        if endpoint is None:
            try:
                resp = self.cf.describe_stacks(StackName=self.stack_name)
                outputs = {o["OutputKey"]: o["OutputValue"] 
                          for o in resp["Stacks"][0].get("Outputs", [])}
                endpoint = outputs.get("ApiEndpoint")
            except ClientError:
                pass
        
        if not endpoint:
            print("ERROR: No API endpoint available")
            print("Deploy the stack first: python scripts/setup_aws.py --action deploy-stack")
            return False
        
        print(f"API Endpoint: {endpoint}")
        
        import urllib.request
        import urllib.error
        
        # Test health endpoint
        print("\n1. Testing GET /health...")
        try:
            req = urllib.request.Request(f"{endpoint}/health", method="GET")
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode()
                print(f"   Status: {resp.status} OK")
                print(f"   Response: {body}")
        except urllib.error.HTTPError as e:
            print(f"   HTTP Error {e.code}: {e.read().decode()}")
        except urllib.error.URLError as e:
            print(f"   Connection Error: {e.reason}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test predict endpoint
        print("\n2. Testing POST /predict...")
        try:
            test_data = json.dumps({
                "user_id": "test_user",
                "heart_rate": 72,
                "steps": 5000,
                "timestamp": "2026-01-21T12:00:00Z"
            }).encode()
            
            req = urllib.request.Request(
                f"{endpoint}/predict",
                data=test_data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode()
                print(f"   Status: {resp.status} OK")
                print(f"   Response: {body}")
        except urllib.error.HTTPError as e:
            print(f"   HTTP Error {e.code}: {e.read().decode()}")
        except urllib.error.URLError as e:
            print(f"   Connection Error: {e.reason}")
        except Exception as e:
            print(f"   Error: {e}")
        
        print("\n" + "=" * 60)
        print("API TESTS COMPLETE")
        print("=" * 60)
        
        return True
    
    def run_all(self) -> bool:
        """Run complete setup."""
        print("=" * 60)
        print("CBT PREDICTION PLATFORM - AWS SETUP")
        print("=" * 60)
        print()
        
        # Step 1: Create bucket
        print("STEP 1/5: Create S3 Bucket")
        print("-" * 40)
        if not self.create_bucket():
            print("\nFailed to create bucket. Aborting.")
            return False
        print()
        
        # Step 2: Upload model
        print("STEP 2/5: Upload Model")
        print("-" * 40)
        self.upload_model()  # Don't fail if model not found
        print()
        
        # Step 3: Deploy stack
        print("STEP 3/5: Deploy CloudFormation Stack")
        print("-" * 40)
        outputs = self.deploy_stack()
        if not outputs:
            print("\nERROR: Stack deployment failed. Check the errors above.")
            return False
        print()
        
        # Step 4: Update Lambda
        print("STEP 4/5: Update Lambda Code")
        print("-" * 40)
        if not self.update_lambda():
            print("\nWARNING: Lambda update failed, but API may still work with placeholder.")
        print()
        
        # Step 5: Test API
        print("STEP 5/5: Test API")
        print("-" * 40)
        time.sleep(5)  # Give Lambda a moment
        self.test_api(outputs.get("ApiEndpoint"))
        
        print()
        print("=" * 60)
        print("SETUP COMPLETE!")
        print("=" * 60)
        
        if outputs.get("ApiEndpoint"):
            print(f"\nYour API is ready at:")
            print(f"  {outputs['ApiEndpoint']}")
            print(f"\nTest it with:")
            print(f"  curl {outputs['ApiEndpoint']}/health")
            print(f"  curl -X POST {outputs['ApiEndpoint']}/predict -H 'Content-Type: application/json' -d '{{\"heart_rate\": 72}}'")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="AWS setup for CBT Prediction Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/setup_aws.py --action all
  python scripts/setup_aws.py --action deploy-stack
  python scripts/setup_aws.py --action test
        """
    )
    parser.add_argument(
        "--action", 
        choices=["all", "create-bucket", "upload-model", "deploy-stack", "update-lambda", "package", "test"],
        default="all",
        help="Action to perform"
    )
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--project-name", default="cbt-prediction", help="Project name")
    parser.add_argument("--model-dir", default=None, help="Model directory path")
    
    args = parser.parse_args()
    
    global MODELS_DIR
    if args.model_dir:
        MODELS_DIR = Path(args.model_dir)
    
    setup = AWSSetup(region=args.region, project_name=args.project_name)
    
    actions = {
        "all": setup.run_all,
        "create-bucket": setup.create_bucket,
        "upload-model": setup.upload_model,
        "deploy-stack": setup.deploy_stack,
        "update-lambda": setup.update_lambda,
        "package": setup.create_lambda_package,
        "test": setup.test_api
    }
    
    result = actions[args.action]()
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()