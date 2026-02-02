# Docker Setup for Local Lambda Testing

This guide explains how to run and test the CBT Prediction API locally using Docker.

## Prerequisites

- **Docker Desktop** installed and running
- **Python 3.11+** for running test scripts
- Trained model files in `models/study_model_v1/`

## Quick Start

### 1. Start the API

```powershell
# Build and start the Lambda container
docker-compose up --build
```

The API will be available at `http://localhost:9000`

### 2. Test the API

In a **new terminal** (keep docker-compose running):

```powershell
# Activate your Python environment
conda activate cbt  # or: venv\Scripts\activate

# Health check
python test_local_api.py --health

# Run all tests
python test_local_api.py --all

# Test specific scenarios
python test_local_api.py --resting
python test_local_api.py --active
python test_local_api.py --sleep
```

## Architecture

### Docker Configuration

```
cbt_pred_platform/
├── Dockerfile                  # AWS Lambda Python 3.11 image
├── docker-compose.yml          # Service configuration
├── requirements-lambda.txt     # Lambda dependencies
└── models/study_model_v1/      # Model files (mounted as volume)
    ├── model.joblib
    ├── scaler.joblib
    └── feature_names.json
```

### How It Works

1. **Base Image**: Uses AWS Lambda Python 3.11 runtime
2. **Volume Mounting**: Model files are mounted read-only from local disk
3. **Environment**: `LOCAL_MODEL_DIR=/var/task/models` tells handler to load locally
4. **Port Mapping**: Container port 8080 → Host port 9000
5. **Hot Reload**: Watch mode rebuilds on code changes

## API Endpoints

### Health Check

**Request:**
```bash
curl -X POST http://localhost:9000/2015-03-31/functions/function/invocations \
  -H "Content-Type: application/json" \
  -d '{"rawPath": "/health"}'
```

**Response:**
```json
{
  "statusCode": 200,
  "body": {
    "status": "healthy",
    "version": "1.0.0",
    "model_loaded": true,
    "timestamp": "2026-02-02T07:30:00.000Z"
  }
}
```

### Predict with Features

**Request:**
```bash
curl -X POST http://localhost:9000/2015-03-31/functions/function/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "rawPath": "/predict",
    "requestContext": {"http": {"method": "POST"}},
    "body": "{\"features\": {\"temperature\": 33.2, \"bpm\": 68, ...}}"
  }'
```

**Response:**
```json
{
  "statusCode": 200,
  "body": {
    "prediction": {
      "cbt_celsius": 36.85,
      "cbt_fahrenheit": 98.33
    },
    "features_used": 84,
    "timestamp": "2026-02-02T07:30:00.000Z"
  }
}
```

### Predict from Time-Series

**Request:**
```json
{
  "rawPath": "/predict",
  "requestContext": {"http": {"method": "POST"}},
  "body": {
    "data": {
      "heart_rate": [
        {"timestamp": "2026-02-02T07:25:00Z", "value": 68},
        {"timestamp": "2026-02-02T07:26:00Z", "value": 69}
      ],
      "skin_temp": [...],
      "room_temp": [...],
      "room_humidity": [...]
    },
    "timestamp": "2026-02-02T07:30:00Z"
  }
}
```

## Testing Guide

### Using Python Test Script

The recommended way to test:

```powershell
# Show help
python test_local_api.py --help

# Health check only (default)
python test_local_api.py

# Run all tests
python test_local_api.py --all

# Individual tests
python test_local_api.py --health
python test_local_api.py --resting
python test_local_api.py --active
python test_local_api.py --sleep

# Debug feature validation
python test_local_api.py --debug
```

### Test Scenarios

The test script includes 3 physiological scenarios:

1. **Resting State**: HR=68, Skin Temp=33.2°C
2. **Active State**: HR=95, Skin Temp=34.5°C
3. **Sleep State**: HR=55, Skin Temp=32.5°C

Each generates 84 features matching the model's training format.

### Expected Output

```
======================================================================
HEALTH CHECK
======================================================================
Status: healthy
Model Loaded: True
✅ Model loaded successfully!

======================================================================
PREDICTION: Resting State (Morning)
======================================================================
Input: HR=68, Skin Temp=33.2°C, Env Temp=21.5°C, Humidity=45%
✅ Predicted CBT: 36.85°C (98.33°F)
```

## Troubleshooting

### Model Not Loading

**Problem:** `model_loaded: false` in health check

**Solutions:**

1. Check model files exist:
   ```powershell
   dir models\study_model_v1
   # Should show: model.joblib, scaler.joblib, feature_names.json
   ```

2. Check container can access files:
   ```powershell
   docker exec cbt_pred_platform-cbt-api-1 ls -la /var/task/models/
   ```

3. Check logs for errors:
   ```powershell
   docker logs cbt_pred_platform-cbt-api-1
   ```

### Missing Features Error

**Problem:** `Missing features: ['feature_names', 'n_features']`

**Solution:** This was a bug in handler.py that's now fixed. Rebuild:
```powershell
docker-compose down
docker-compose up --build
```

### Connection Refused

**Problem:** `Connection failed` in test script

**Solutions:**

1. Ensure Docker Desktop is running
2. Check container is running: `docker ps`
3. Verify port mapping: Should show `0.0.0.0:9000->8080/tcp`

### Feature Count Mismatch

**Problem:** Wrong number of features sent

**Solution:** Use the debug script to check:
```powershell
python check_model_features.py
```

This shows exactly what 84 features the model expects.

## Development Workflow

### Making Code Changes

The Docker setup supports **watch mode** for automatic rebuilds:

```powershell
# Start with watch enabled
docker-compose up --build --watch

# Or press 'w' in the running docker-compose terminal
```

Watch rebuilds when these change:
- `src/serving/`
- `src/features/`
- `requirements-lambda.txt`

### Viewing Logs

```powershell
# Live logs (follow mode)
docker logs -f cbt_pred_platform-cbt-api-1

# Last 50 lines
docker logs --tail 50 cbt_pred_platform-cbt-api-1

# Search for errors
docker logs cbt_pred_platform-cbt-api-1 2>&1 | Select-String "error|Error"
```

### Stopping the Service

```powershell
# Stop (keeps containers)
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop, remove, and clean up volumes
docker-compose down -v
```

## Advanced Usage

### Custom Environment Variables

Edit `docker-compose.yml`:

```yaml
environment:
  - LOCAL_MODEL_DIR=/var/task/models
  - LOG_LEVEL=DEBUG
  - MODEL_VERSION=v1
```

### Using Different Models

```yaml
volumes:
  - ./models/my_other_model:/var/task/models:ro
```

### Running in Background

```powershell
docker-compose up -d --build
```

### Accessing Container Shell

```powershell
docker exec -it cbt_pred_platform-cbt-api-1 /bin/bash
```

## Model Requirements

The Lambda function expects these files in the model directory:

| File | Required | Purpose |
|------|----------|---------|
| `model.joblib` | ✅ Yes | XGBoost model (pickled) |
| `scaler.joblib` | ✅ Yes | StandardScaler for features |
| `feature_names.json` | ✅ Yes | List of 84 feature names in order |
| `metadata.json` | ⚠️ Optional | Training metadata |

### Feature Names Format

```json
{
  "feature_names": [
    "temperature",
    "temp_mean_5min",
    "temp_median_5min",
    ...
  ],
  "n_features": 84
}
```

## Integration with Production

This local setup mirrors AWS Lambda exactly:

- Same Python runtime (3.11)
- Same file structure
- Same environment variables
- Same handler function

**To deploy to AWS:**

1. Build deployment package:
   ```powershell
   .\scripts\package_lambda.ps1
   ```

2. Upload to S3 and configure Lambda:
   - Set `MODEL_BUCKET` environment variable
   - Remove `LOCAL_MODEL_DIR`
   - Model loads from S3 instead

## Performance Notes

- **Cold Start**: ~1.5 seconds (model loading)
- **Warm Start**: ~20-50ms (model cached)
- **Container Memory**: 3008 MB (matches Lambda)
- **Watch Rebuild**: ~5-10 seconds

## References

- [AWS Lambda Python Runtime](https://docs.aws.amazon.com/lambda/latest/dg/python-image.html)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Lambda Container Images](https://docs.aws.amazon.com/lambda/latest/dg/images-create.html)