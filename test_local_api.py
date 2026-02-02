"""Test the local Docker Lambda API."""
import argparse
import requests
import json
import sys

URL = "http://localhost:9000/2015-03-31/functions/function/invocations"

# Load the actual feature names the model expects
def load_expected_features():
    """Load feature names from the model."""
    try:
        with open("models/study_model_v1/feature_names.json") as f:
            data = json.load(f)
            return data.get("feature_names", data)
    except Exception as e:
        print(f"Warning: Could not load feature names: {e}")
        return None

EXPECTED_FEATURES = load_expected_features()

def get_base_features(bpm: float, skin_temp: float, env_temp: float, humidity: float) -> dict:
    """
    Generate 84 features using the exact names the model expects.
    
    Uses feature_names.json to ensure compatibility.
    """
    if EXPECTED_FEATURES is None:
        print("ERROR: Could not load feature names from model")
        return {}
    
    # Create features dict - these are the BASE VALUES
    # The model expects specific feature names from training
    features = {}
    
    # Assign the 4 current values (these match training column names)
    # Check your actual feature_names.json for exact names
    features["bpm"] = bpm
    features["temperature"] = skin_temp
    features["env_Temperature_Celsius"] = env_temp
    features["Relative_Humidity"] = humidity
    
    # Rolling stats for heart rate
    features["bpm_mean_5min"] = bpm - 0.5
    features["bpm_median_5min"] = bpm
    features["bpm_std_5min"] = 2.1
    features["bpm_mean_20min"] = bpm - 1.2
    features["bpm_median_20min"] = bpm - 1.0
    features["bpm_std_20min"] = 3.2
    features["bpm_mean_35min"] = bpm - 1.5
    features["bpm_median_35min"] = bpm - 1.0
    features["bpm_std_35min"] = 3.5
    features["bpm_mean_5min_lag10m"] = bpm - 1.0
    features["bpm_median_5min_lag10m"] = bpm - 0.5
    features["bpm_std_5min_lag10m"] = 2.5
    features["bpm_mean_20min_lag10m"] = bpm - 1.5
    features["bpm_median_20min_lag10m"] = bpm - 1.0
    features["bpm_std_20min_lag10m"] = 3.0
    features["bpm_mean_35min_lag10m"] = bpm - 1.8
    features["bpm_median_35min_lag10m"] = bpm - 1.5
    features["bpm_std_35min_lag10m"] = 3.3
    features["bpm_diff_1"] = 0.5
    features["bpm_slope_5m"] = 0.1
    
    # Rolling stats for skin temp
    features["temp_mean_5min"] = skin_temp - 0.1
    features["temp_median_5min"] = skin_temp
    features["temp_std_5min"] = 0.15
    features["temp_mean_20min"] = skin_temp - 0.2
    features["temp_median_20min"] = skin_temp - 0.1
    features["temp_std_20min"] = 0.2
    features["temp_mean_35min"] = skin_temp - 0.3
    features["temp_median_35min"] = skin_temp - 0.2
    features["temp_std_35min"] = 0.25
    features["temp_mean_5min_lag10m"] = skin_temp - 0.2
    features["temp_median_5min_lag10m"] = skin_temp - 0.1
    features["temp_std_5min_lag10m"] = 0.18
    features["temp_mean_20min_lag10m"] = skin_temp - 0.3
    features["temp_median_20min_lag10m"] = skin_temp - 0.2
    features["temp_std_20min_lag10m"] = 0.22
    features["temp_mean_35min_lag10m"] = skin_temp - 0.4
    features["temp_median_35min_lag10m"] = skin_temp - 0.3
    features["temp_std_35min_lag10m"] = 0.28
    features["temp_diff_1"] = 0.1
    features["temp_slope_5m"] = 0.02
    
    # Rolling stats for environmental temp
    features["temp_env_mean_5min"] = env_temp - 0.1
    features["temp_env_median_5min"] = env_temp
    features["temp_env_std_5min"] = 0.1
    features["temp_env_mean_20min"] = env_temp - 0.2
    features["temp_env_median_20min"] = env_temp - 0.1
    features["temp_env_std_20min"] = 0.2
    features["temp_env_mean_35min"] = env_temp - 0.3
    features["temp_env_median_35min"] = env_temp - 0.2
    features["temp_env_std_35min"] = 0.25
    features["temp_env_mean_5min_lag10m"] = env_temp - 0.2
    features["temp_env_median_5min_lag10m"] = env_temp - 0.1
    features["temp_env_std_5min_lag10m"] = 0.15
    features["temp_env_mean_20min_lag10m"] = env_temp - 0.3
    features["temp_env_median_20min_lag10m"] = env_temp - 0.2
    features["temp_env_std_20min_lag10m"] = 0.22
    features["temp_env_mean_35min_lag10m"] = env_temp - 0.4
    features["temp_env_median_35min_lag10m"] = env_temp - 0.3
    features["temp_env_std_35min_lag10m"] = 0.28
    features["temp_env_diff_1"] = 0.1
    features["temp_env_slope_5m"] = 0.05
    
    # Rolling stats for humidity
    features["humidity_env_mean_5min"] = humidity
    features["humidity_env_median_5min"] = humidity
    features["humidity_env_std_5min"] = 1.0
    features["humidity_env_mean_20min"] = humidity - 0.2
    features["humidity_env_median_20min"] = humidity
    features["humidity_env_std_20min"] = 1.5
    features["humidity_env_mean_35min"] = humidity - 0.5
    features["humidity_env_median_35min"] = humidity - 0.2
    features["humidity_env_std_35min"] = 2.0
    features["humidity_env_mean_5min_lag10m"] = humidity - 0.2
    features["humidity_env_median_5min_lag10m"] = humidity
    features["humidity_env_std_5min_lag10m"] = 1.2
    features["humidity_env_mean_20min_lag10m"] = humidity - 0.4
    features["humidity_env_median_20min_lag10m"] = humidity - 0.2
    features["humidity_env_std_20min_lag10m"] = 1.8
    features["humidity_env_mean_35min_lag10m"] = humidity - 0.7
    features["humidity_env_median_35min_lag10m"] = humidity - 0.5
    features["humidity_env_std_35min_lag10m"] = 2.2
    features["humidity_env_diff_1"] = 0.2
    features["humidity_env_slope_5m"] = 0.1
    
    # Verify we have all expected features
    if EXPECTED_FEATURES:
        missing = [f for f in EXPECTED_FEATURES if f not in features]
        if missing:
            print(f"\n⚠️  WARNING: Missing {len(missing)} features:")
            for f in missing[:5]:
                print(f"    - {f}")
            # Fill missing with zeros
            for f in missing:
                features[f] = 0.0
    
    return features

def call_api(payload: dict) -> dict:
    """Make API call and return parsed response."""
    try:
        response = requests.post(URL, json=payload, timeout=30)
        result = response.json()
        
        if "body" in result and isinstance(result["body"], str):
            result["body"] = json.loads(result["body"])
        
        return result
    except requests.exceptions.ConnectionError:
        return {"error": "Connection failed. Is Docker running? Try: docker-compose up --build"}
    except Exception as e:
        return {"error": str(e)}

def test_health() -> bool:
    """Test health endpoint."""
    print("=" * 70)
    print("HEALTH CHECK")
    print("=" * 70)
    
    result = call_api({"rawPath": "/health"})
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return False
    
    body = result.get("body", result)
    print(f"Status: {body.get('status', 'unknown')}")
    print(f"Model Loaded: {body.get('model_loaded', 'unknown')}")
    print(f"Version: {body.get('version', 'unknown')}")
    print(f"Timestamp: {body.get('timestamp', 'unknown')}")
    
    if body.get('model_loaded'):
        print("✅ Model loaded successfully!")
        return True
    else:
        print("⚠️  Model NOT loaded")
        return False

def test_debug() -> None:
    """Debug model loading."""
    print("=" * 70)
    print("DEBUG: Feature Names Check")
    print("=" * 70)
    
    if EXPECTED_FEATURES:
        print(f"\nModel expects {len(EXPECTED_FEATURES)} features:")
        print("\nFirst 10:")
        for i, name in enumerate(EXPECTED_FEATURES[:10], 1):
            print(f"  {i}. {name}")
        
        print(f"\n... and {len(EXPECTED_FEATURES) - 10} more")
    else:
        print("❌ Could not load feature names from model")

def test_predict(scenario: str, bpm: float, skin_temp: float, env_temp: float, humidity: float) -> bool:
    """Test prediction."""
    print("=" * 70)
    print(f"PREDICTION: {scenario}")
    print("=" * 70)
    
    features = get_base_features(bpm, skin_temp, env_temp, humidity)
    print(f"Input: HR={bpm}, Skin Temp={skin_temp}°C, Env Temp={env_temp}°C, Humidity={humidity}%")
    print(f"Features generated: {len(features)}")
    
    payload = {
        "rawPath": "/predict",
        "requestContext": {"http": {"method": "POST"}},
        "body": json.dumps({"features": features})
    }
    
    result = call_api(payload)
    body = result.get("body", result)
    
    if "error" in body:
        print(f"❌ Error: {body['error']}")
        if "missing" in body:
            print(f"   Missing features: {body.get('missing_count', 'unknown')}")
            print(f"   First few missing: {body.get('missing', [])[:5]}")
        return False
    
    if "prediction" in body:
        pred = body["prediction"]
        print(f"✅ Predicted CBT: {pred.get('cbt_celsius', 'N/A')}°C ({pred.get('cbt_fahrenheit', 'N/A')}°F)")
        return True
    
    print(f"⚠️  Unexpected response: {body}")
    return False

def test_resting() -> bool:
    return test_predict("Resting State (Morning)", bpm=68, skin_temp=33.2, env_temp=21.5, humidity=45)

def test_active() -> bool:
    return test_predict("Active State (Exercise)", bpm=95, skin_temp=34.5, env_temp=23.0, humidity=50)

def test_sleep() -> bool:
    return test_predict("Sleep State (Recovery)", bpm=55, skin_temp=32.5, env_temp=20.0, humidity=42)

def run_all_tests() -> None:
    """Run all tests."""
    print("\n" + "=" * 70)
    print("RUNNING ALL TESTS")
    print("=" * 70 + "\n")
    
    results = []
    results.append(("Health Check", test_health()))
    print()
    results.append(("Resting Prediction", test_resting()))
    print()
    results.append(("Active Prediction", test_active()))
    print()
    results.append(("Sleep Prediction", test_sleep()))
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")

def main():
    parser = argparse.ArgumentParser(description="Test CBT Prediction API")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--health", action="store_true", help="Health check")
    parser.add_argument("--predict", action="store_true", help="All predictions")
    parser.add_argument("--resting", action="store_true", help="Resting prediction")
    parser.add_argument("--active", action="store_true", help="Active prediction")
    parser.add_argument("--sleep", action="store_true", help="Sleep prediction")
    parser.add_argument("--debug", action="store_true", help="Debug features")
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        test_health()
        return
    
    if args.debug:
        test_debug()
        return
    
    if args.all:
        run_all_tests()
        return
    
    if args.health:
        test_health()
    if args.predict:
        test_resting()
        print()
        test_active()
        print()
        test_sleep()
    if args.resting:
        test_resting()
    if args.active:
        test_active()
    if args.sleep:
        test_sleep()

if __name__ == "__main__":
    main()