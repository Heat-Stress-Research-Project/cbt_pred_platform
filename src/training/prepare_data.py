"""
Data Preparation for Training

Loads all data sources, aligns features to CBT measurements,
and creates the training dataset.

KEY CONCEPT:
    Each CBT measurement becomes ONE training sample.
    Features are computed from the time window BEFORE the measurement.
    
    Example:
        CBT measurement at 9:30 PM = 98.7°F (37.06°C)
        Features computed from 9:20 PM to 9:30 PM window (10 min)
        
        X = [temperature, temp_mean_5min, ..., bpm, bpm_mean_5min, ...]
        y = 37.06

ALIGNMENT:
    - CBT timestamps are the anchor points
    - Feature window: 10 minutes before each CBT measurement
    - Alignment tolerance: ±15 minutes for matching sensor data

REQUIREMENTS:
    All four signals (skin_temperature, heart_rate, ambient_temp, humidity)
    are REQUIRED for valid training samples. No minimum sample counts.

Total Features: 86 (84 training + 2 metadata: user_id, timestamp)
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import warnings

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.transformations import (
    FeatureTransformer, 
    DataLoader, 
    align_data_to_target,
)

# Progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        """Fallback for tqdm if not installed."""
        desc = kwargs.get('desc', '')
        if desc:
            print(f"{desc}...")
        return iterable


class DataPreparer:
    """
    Prepares training data by aligning features to CBT measurements.
    
    Usage:
        preparer = DataPreparer(data_dir="data/raw")
        X, y, metadata = preparer.prepare()
        
    Data Requirements:
        - Fitbit heart rate CSV (timestamp, beats per minute)
        - Fitbit skin temperature CSV (recorded_time, temperature)
        - Govee environmental CSV (Timestamp, Temperature_Fahrenheit, Relative_Humidity)
        - CBT labels CSV (Date, Time, CBT (Deg F))
    
    ALL FOUR SIGNALS ARE REQUIRED for each sample.
    
    Alignment:
        - Feature window: 10 minutes (configurable)
        - Alignment tolerance: 15 minutes (configurable)
        - Anchor: CBT measurement timestamps
    """
    
    # Default configuration
    DEFAULT_FEATURE_WINDOW_MINUTES = 10
    DEFAULT_ALIGNMENT_TOLERANCE_MINUTES = 15  # Increased from 5
    
    def __init__(
        self,
        data_dir: Union[str, Path] = None,
        feature_window_minutes: int = DEFAULT_FEATURE_WINDOW_MINUTES,
        alignment_tolerance_minutes: int = DEFAULT_ALIGNMENT_TOLERANCE_MINUTES,
        user_id: str = "default_user"
    ):
        """
        Initialize DataPreparer.
        
        Args:
            data_dir: Directory containing raw data files
            feature_window_minutes: How many minutes of data to use for features (default: 10)
            alignment_tolerance_minutes: Tolerance for matching sensor data to CBT time (default: 15)
            user_id: User identifier for the data
        """
        self.data_dir = Path(data_dir) if data_dir else Path("data/raw")
        self.feature_window_minutes = feature_window_minutes
        self.alignment_tolerance_minutes = alignment_tolerance_minutes
        self.user_id = user_id
        self.transformer = FeatureTransformer()
        
        # Storage for loaded data
        self.fitbit_hr: Optional[pd.DataFrame] = None
        self.fitbit_skin_temp: Optional[pd.DataFrame] = None
        self.fitbit_combined: Optional[pd.DataFrame] = None
        self.env_data: Optional[pd.DataFrame] = None
        self.cbt_labels: Optional[pd.DataFrame] = None
        
    def load_fitbit_heart_rate(
        self, 
        filepath: Union[str, Path] = None
    ) -> pd.DataFrame:
        """
        Load Fitbit heart rate data.
        
        Expected format:
            timestamp, beats per minute
            2025-04-10T00:00:01Z, 67
        
        Args:
            filepath: Path to heart rate CSV file.
        
        Returns:
            DataFrame with heart rate data
        """
        if filepath is None:
            for pattern in ["*heart*rate*.csv", "*hr*.csv", "*fitbit*hr*.csv"]:
                files = list(self.data_dir.glob(pattern))
                if files:
                    filepath = files[0]
                    break
        
        if filepath is None:
            raise FileNotFoundError(
                f"No Fitbit heart rate data found in {self.data_dir}. "
                "Expected file with 'heart_rate' or 'hr' in name."
            )
        
        filepath = Path(filepath)
        print(f"Loading Fitbit heart rate from: {filepath.name}")
        
        df = DataLoader.load_fitbit_hr(str(filepath), user_id=self.user_id)
        
        self.fitbit_hr = df
        print(f"  ✓ Loaded {len(df):,} rows")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  HR range: {df['heart_rate'].min():.0f} - {df['heart_rate'].max():.0f} bpm")
        
        return df
    
    def load_fitbit_skin_temperature(
        self, 
        filepath: Union[str, Path] = None
    ) -> pd.DataFrame:
        """
        Load Fitbit wrist/skin temperature data.
        
        Expected format:
            recorded_time, temperature
            2025-05-30T00:00, -2.095291443
        
        Note: Temperature values are deviations from baseline.
        
        Args:
            filepath: Path to skin temperature CSV file.
        
        Returns:
            DataFrame with skin temperature data
        """
        if filepath is None:
            for pattern in ["*skin*temp*.csv", "*wrist*temp*.csv", "*temperature*.csv"]:
                files = list(self.data_dir.glob(pattern))
                # Exclude files that look like environmental data
                files = [f for f in files if "env" not in f.name.lower() and "govee" not in f.name.lower()]
                if files:
                    filepath = files[0]
                    break
        
        if filepath is None:
            raise FileNotFoundError(
                f"No Fitbit skin temperature data found in {self.data_dir}. "
                "Expected file with 'skin_temp' or 'wrist_temp' in name."
            )
        
        filepath = Path(filepath)
        print(f"Loading Fitbit skin temperature from: {filepath.name}")
        
        df = DataLoader.load_fitbit_skin_temp(str(filepath), user_id=self.user_id)
        
        self.fitbit_skin_temp = df
        print(f"  ✓ Loaded {len(df):,} rows")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Temp deviation range: {df['skin_temperature'].min():.2f} to {df['skin_temperature'].max():.2f}")
        
        return df
    
    def load_environmental_data(
        self, 
        filepath: Union[str, Path] = None
    ) -> pd.DataFrame:
        """
        Load Govee environmental sensor data.
        
        Expected format:
            Timestamp for sample frequency every 1 min min, PM2.5(µg/m³), Temperature_Fahrenheit, Relative_Humidity
            2025-04-07 16:51:00, 0, 69.98, 50.6
        
        Processing:
            - Drops PM2.5 column
            - Converts temperature from Fahrenheit to Celsius
        
        Args:
            filepath: Path to environmental data CSV file.
        
        Returns:
            DataFrame with environmental data
        """
        if filepath is None:
            for pattern in ["*govee*.csv", "*env*.csv", "*ambient*.csv"]:
                files = list(self.data_dir.glob(pattern))
                if files:
                    filepath = files[0]
                    break
        
        if filepath is None:
            raise FileNotFoundError(
                f"No environmental data found in {self.data_dir}. "
                "Expected file with 'govee', 'env', or 'ambient' in name."
            )
        
        filepath = Path(filepath)
        print(f"Loading environmental data from: {filepath.name}")
        
        df = DataLoader.load_govee_env(str(filepath), user_id=self.user_id)
        
        self.env_data = df
        print(f"  ✓ Loaded {len(df):,} rows")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Temp range: {df['ambient_temp'].min():.1f}°C to {df['ambient_temp'].max():.1f}°C")
        print(f"  Humidity range: {df['humidity'].min():.1f}% to {df['humidity'].max():.1f}%")
        
        return df
    
    def load_cbt_labels(
        self, 
        filepath: Union[str, Path] = None
    ) -> pd.DataFrame:
        """
        Load CBT (Core Body Temperature) measurements - these are the labels (y).
        
        Expected format:
            Date:, Time:, CBT (Deg F):
            4/5/2025, 8:59 PM, 98.5
        
        Processing:
            - Combines Date and Time into timestamp
            - Converts from Central Time to UTC
            - Converts from Fahrenheit to Celsius
        
        Args:
            filepath: Path to CBT data CSV file.
        
        Returns:
            DataFrame with CBT measurements
        """
        if filepath is None:
            for pattern in ["*cbt*.csv", "*core*body*.csv", "*braun*.csv", "*ear*.csv"]:
                files = list(self.data_dir.glob(pattern))
                if files:
                    filepath = files[0]
                    break
        
        if filepath is None:
            raise FileNotFoundError(
                f"No CBT label data found in {self.data_dir}. "
                "Expected file with 'cbt', 'core_body', or 'braun' in name."
            )
        
        filepath = Path(filepath)
        print(f"Loading CBT labels from: {filepath.name}")
        
        df = DataLoader.load_cbt(str(filepath), user_id=self.user_id)
        
        # Validation
        if df["cbt_celsius"].min() < 35 or df["cbt_celsius"].max() > 42:
            warnings.warn(
                f"CBT values outside normal range: "
                f"{df['cbt_celsius'].min():.2f}°C - {df['cbt_celsius'].max():.2f}°C"
            )
        
        self.cbt_labels = df
        print(f"  ✓ Loaded {len(df)} CBT measurements")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  CBT range: {df['cbt_celsius'].min():.2f}°C to {df['cbt_celsius'].max():.2f}°C")
        
        return df
    
    def combine_fitbit_data(self) -> pd.DataFrame:
        """
        Combine heart rate and skin temperature into single DataFrame.
        
        Merges on nearest timestamp within alignment tolerance.
        
        Returns:
            Combined DataFrame with both heart_rate and skin_temperature
        """
        if self.fitbit_hr is None:
            raise ValueError("Heart rate data not loaded. Call load_fitbit_heart_rate() first.")
        if self.fitbit_skin_temp is None:
            raise ValueError("Skin temperature data not loaded. Call load_fitbit_skin_temperature() first.")
        
        print("\nCombining Fitbit data...")
        
        tolerance = pd.Timedelta(minutes=self.alignment_tolerance_minutes)
        
        # Merge on nearest timestamp
        combined = pd.merge_asof(
            self.fitbit_hr.sort_values("timestamp"),
            self.fitbit_skin_temp[["timestamp", "skin_temperature"]].sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=tolerance
        )
        
        # Report merge quality
        hr_count = combined["heart_rate"].notna().sum()
        skin_count = combined["skin_temperature"].notna().sum()
        both_count = (combined["heart_rate"].notna() & combined["skin_temperature"].notna()).sum()
        
        print(f"  ✓ Combined rows: {len(combined):,}")
        print(f"  Heart rate values: {hr_count:,}")
        print(f"  Skin temp values: {skin_count:,}")
        print(f"  Both available: {both_count:,}")
        print(f"  Alignment tolerance: ±{self.alignment_tolerance_minutes} minutes")
        
        self.fitbit_combined = combined
        return combined
    
    def _find_nearest_data_window(
        self,
        target_time: pd.Timestamp,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp"
    ) -> Tuple[pd.Timestamp, pd.Timestamp, bool]:
        """
        Find the best data window around the target time.
        
        Uses a sliding window approach:
        1. First try exact window [target - window_size, target]
        2. If no data, slide ±tolerance to find ANY data
        
        Args:
            target_time: CBT measurement timestamp
            df: DataFrame with sensor data
            timestamp_col: Name of timestamp column
        
        Returns:
            (window_start, window_end, found_data)
        """
        window_size = pd.Timedelta(minutes=self.feature_window_minutes)
        
        # Exact window
        window_end = target_time
        window_start = target_time - window_size
        
        # Check if we have ANY data in this window
        mask = (df[timestamp_col] >= window_start) & (df[timestamp_col] <= window_end)
        if mask.sum() > 0:
            return window_start, window_end, True
        
        # Slide window within tolerance to find data
        for offset_minutes in range(1, self.alignment_tolerance_minutes + 1):
            # Try shifting forward
            shifted_end = target_time + pd.Timedelta(minutes=offset_minutes)
            shifted_start = shifted_end - window_size
            mask = (df[timestamp_col] >= shifted_start) & (df[timestamp_col] <= shifted_end)
            if mask.sum() > 0:
                return shifted_start, shifted_end, True
            
            # Try shifting backward
            shifted_end = target_time - pd.Timedelta(minutes=offset_minutes)
            shifted_start = shifted_end - window_size
            mask = (df[timestamp_col] >= shifted_start) & (df[timestamp_col] <= shifted_end)
            if mask.sum() > 0:
                return shifted_start, shifted_end, True
        
        return window_start, window_end, False
    
    def _check_window_has_all_signals(
        self,
        fitbit_window: pd.DataFrame,
        env_window: pd.DataFrame
    ) -> Tuple[bool, str]:
        """
        Check if window has ALL required signals (at least 1 sample each).
        
        ALL FOUR signals are required:
            - heart_rate
            - skin_temperature
            - ambient_temp
            - humidity
        
        Args:
            fitbit_window: DataFrame with heart_rate and skin_temperature
            env_window: DataFrame with ambient_temp and humidity
        
        Returns:
            (is_valid, reason_if_invalid)
        """
        # Check heart rate - need at least 1
        hr_count = fitbit_window["heart_rate"].notna().sum()
        if hr_count == 0:
            return False, "heart_rate: no data"
        
        # Check skin temperature - need at least 1
        skin_count = fitbit_window["skin_temperature"].notna().sum()
        if skin_count == 0:
            return False, "skin_temperature: no data"
        
        # Check ambient temperature - need at least 1
        if "ambient_temp" not in env_window.columns:
            return False, "ambient_temp: column missing"
        ambient_count = env_window["ambient_temp"].notna().sum()
        if ambient_count == 0:
            return False, "ambient_temp: no data"
        
        # Check humidity - need at least 1
        if "humidity" not in env_window.columns:
            return False, "humidity: column missing"
        humidity_count = env_window["humidity"].notna().sum()
        if humidity_count == 0:
            return False, "humidity: no data"
        
        return True, ""
    
    def align_features_to_labels(self) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """
        Align features to each CBT measurement timestamp.
        
        For each CBT reading:
            1. Get the measurement timestamp (anchor point)
            2. Find best 10-minute window (with ±15min tolerance)
            3. Extract Fitbit data from window
            4. Extract env data from same window
            5. Validate ALL 4 signals have at least 1 sample
            6. Compute 84 features using FeatureTransformer
            7. Store features as X, CBT as y
        
        Returns:
            X: DataFrame of 86 features (84 training + 2 metadata)
            y: Series of CBT values (Celsius)
            metadata: Dict with alignment info
        """
        if self.fitbit_combined is None:
            raise ValueError("Fitbit data not combined. Call combine_fitbit_data() first.")
        if self.env_data is None:
            raise ValueError("Environmental data not loaded. Call load_environmental_data() first.")
        if self.cbt_labels is None:
            raise ValueError("CBT labels not loaded. Call load_cbt_labels() first.")
        
        n_measurements = len(self.cbt_labels)
        
        print(f"\n{'='*60}")
        print("ALIGNING FEATURES TO CBT MEASUREMENTS")
        print(f"{'='*60}")
        print(f"  CBT measurements: {n_measurements}")
        print(f"  Feature window: {self.feature_window_minutes} minutes")
        print(f"  Alignment tolerance: ±{self.alignment_tolerance_minutes} minutes")
        print(f"  Required signals: heart_rate, skin_temperature, ambient_temp, humidity")
        print(f"  Minimum samples: 1 per signal (any data is valid)")
        print()
        
        X_list = []
        y_list = []
        timestamps_used = []
        skipped = {
            "no_fitbit_data": [],
            "no_env_data": [],
            "missing_signals": [],
            "transform_error": []
        }
        
        # Progress bar for alignment
        pbar = tqdm(
            self.cbt_labels.iterrows(),
            total=n_measurements,
            desc="Aligning features",
            unit="samples",
            disable=not HAS_TQDM
        )
        
        for idx, row in pbar:
            measurement_time = row["timestamp"]
            cbt_value = row["cbt_celsius"]
            
            # Update progress bar description
            if HAS_TQDM:
                pbar.set_postfix({
                    "valid": len(X_list),
                    "skipped": sum(len(v) for v in skipped.values())
                })
            
            # Find best Fitbit window
            fitbit_start, fitbit_end, fitbit_found = self._find_nearest_data_window(
                measurement_time, self.fitbit_combined, "timestamp"
            )
            
            if not fitbit_found:
                skipped["no_fitbit_data"].append(str(measurement_time))
                continue
            
            # Find best environmental window
            env_start, env_end, env_found = self._find_nearest_data_window(
                measurement_time, self.env_data, "timestamp"
            )
            
            if not env_found:
                skipped["no_env_data"].append(str(measurement_time))
                continue
            
            # Extract data windows
            fitbit_window = self.fitbit_combined[
                (self.fitbit_combined["timestamp"] >= fitbit_start) &
                (self.fitbit_combined["timestamp"] <= fitbit_end)
            ].copy()
            
            env_window = self.env_data[
                (self.env_data["timestamp"] >= env_start) &
                (self.env_data["timestamp"] <= env_end)
            ].copy()
            
            # Validate all 4 signals have at least 1 sample
            is_valid, reason = self._check_window_has_all_signals(fitbit_window, env_window)
            if not is_valid:
                skipped["missing_signals"].append(f"{measurement_time}: {reason}")
                continue
            
            # Compute features
            try:
                features = self.transformer.transform(
                    fitbit_df=fitbit_window,
                    env_df=env_window,
                    target_timestamp=measurement_time,
                    user_id=self.user_id
                )
                
                if features is None or len(features) == 0:
                    skipped["missing_signals"].append(f"{measurement_time}: transform returned empty")
                    continue
                
                X_list.append(features)
                y_list.append(cbt_value)
                timestamps_used.append(measurement_time)
                
            except Exception as e:
                skipped["transform_error"].append({
                    "timestamp": str(measurement_time),
                    "error": str(e)
                })
                continue
        
        pbar.close()
        
        if len(X_list) == 0:
            # Print diagnostic info
            print("\n❌ ERROR: No valid training samples created!")
            print("\nSkipped samples breakdown:")
            for reason, items in skipped.items():
                print(f"  {reason}: {len(items)}")
                if items and len(items) <= 5:
                    for item in items[:5]:
                        print(f"    - {item}")
            
            print("\nDiagnostic info:")
            print(f"  Fitbit data range: {self.fitbit_combined['timestamp'].min()} to {self.fitbit_combined['timestamp'].max()}")
            print(f"  Env data range: {self.env_data['timestamp'].min()} to {self.env_data['timestamp'].max()}")
            print(f"  CBT timestamps range: {self.cbt_labels['timestamp'].min()} to {self.cbt_labels['timestamp'].max()}")
            
            raise ValueError(
                "No valid training samples created. "
                "Check that all data sources overlap in time."
            )
        
        # Combine all features
        print("\nCombining features...")
        X = pd.concat(X_list, ignore_index=True)
        y = pd.Series(y_list, name="cbt_celsius")
        
        # Check for NaN values in training features only
        training_cols = FeatureTransformer.get_training_features()
        available_training_cols = [c for c in training_cols if c in X.columns]
        training_X = X[available_training_cols]
        nan_cols = training_X.columns[training_X.isna().any()].tolist()
        
        if nan_cols:
            print(f"  ⚠ NaN values in {len(nan_cols)} columns, filling with medians...")
            for col in tqdm(nan_cols, desc="  Filling NaN", disable=not HAS_TQDM):
                median_val = X[col].median()
                if pd.isna(median_val):
                    median_val = 0  # Fallback for all-NaN columns
                X[col] = X[col].fillna(median_val)
        
        # Calculate skip statistics
        total_skipped = sum(len(v) for v in skipped.values())
        
        metadata = {
            "n_samples": len(y),
            "n_cbt_measurements": n_measurements,
            "n_features": len(X.columns),
            "n_training_features": len(available_training_cols),
            "feature_names": list(X.columns),
            "training_feature_names": available_training_cols,
            "alignment_config": {
                "feature_window_minutes": self.feature_window_minutes,
                "alignment_tolerance_minutes": self.alignment_tolerance_minutes,
                "required_signals": ["heart_rate", "skin_temperature", "ambient_temp", "humidity"],
                "min_samples_per_signal": 1
            },
            "skip_statistics": {
                "total_skipped": total_skipped,
                "no_fitbit_data": len(skipped["no_fitbit_data"]),
                "no_env_data": len(skipped["no_env_data"]),
                "missing_signals": len(skipped["missing_signals"]),
                "transform_error": len(skipped["transform_error"])
            },
            "cbt_stats": {
                "mean": float(y.mean()),
                "std": float(y.std()),
                "min": float(y.min()),
                "max": float(y.max())
            },
            "user_id": self.user_id,
            "timestamps_used": [str(t) for t in timestamps_used[:10]]  # First 10 for reference
        }
        
        # Print summary
        print()
        print(f"{'='*60}")
        print("ALIGNMENT COMPLETE")
        print(f"{'='*60}")
        print(f"  ✓ Valid samples: {len(y)} / {n_measurements} ({100*len(y)/n_measurements:.1f}%)")
        print(f"  ✓ Features per sample: {len(X.columns)}")
        print(f"    - Training features: {len(available_training_cols)}")
        print(f"    - Metadata features: {len(X.columns) - len(available_training_cols)}")
        print()
        print(f"  Skipped samples: {total_skipped}")
        print(f"    - No Fitbit data: {len(skipped['no_fitbit_data'])}")
        print(f"    - No environmental data: {len(skipped['no_env_data'])}")
        print(f"    - Missing signals: {len(skipped['missing_signals'])}")
        print(f"    - Transform errors: {len(skipped['transform_error'])}")
        print()
        print(f"  CBT Statistics:")
        print(f"    Mean: {y.mean():.2f}°C")
        print(f"    Std:  {y.std():.2f}°C")
        print(f"    Range: {y.min():.2f}°C - {y.max():.2f}°C")
        
        return X, y, metadata
    
    def prepare(
        self,
        fitbit_hr_path: Union[str, Path] = None,
        fitbit_skin_temp_path: Union[str, Path] = None,
        env_path: Union[str, Path] = None,
        cbt_path: Union[str, Path] = None
    ) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """
        Full preparation pipeline.
        
        Args:
            fitbit_hr_path: Path to Fitbit heart rate CSV (optional, auto-detect)
            fitbit_skin_temp_path: Path to Fitbit skin temperature CSV (optional, auto-detect)
            env_path: Path to Govee environmental CSV (optional, auto-detect)
            cbt_path: Path to CBT labels CSV (optional, auto-detect)
        
        Returns:
            X: DataFrame with 86 features
            y: Series with CBT values (Celsius)
            metadata: Dict with preparation info
        """
        print("=" * 60)
        print("DATA PREPARATION PIPELINE")
        print("=" * 60)
        print(f"  Data directory: {self.data_dir}")
        print(f"  User ID: {self.user_id}")
        print(f"  Feature window: {self.feature_window_minutes} minutes")
        print(f"  Alignment tolerance: ±{self.alignment_tolerance_minutes} minutes")
        print(f"  Required signals: ALL (heart_rate, skin_temp, ambient_temp, humidity)")
        print()
        
        # Load all data sources with progress indication
        print("Step 1/5: Loading Fitbit heart rate data...")
        self.load_fitbit_heart_rate(fitbit_hr_path)
        print()
        
        print("Step 2/5: Loading Fitbit skin temperature data...")
        self.load_fitbit_skin_temperature(fitbit_skin_temp_path)
        print()
        
        print("Step 3/5: Loading environmental data...")
        self.load_environmental_data(env_path)
        print()
        
        print("Step 4/5: Loading CBT labels...")
        self.load_cbt_labels(cbt_path)
        
        # Combine Fitbit data
        print("\nStep 5/5: Processing and aligning data...")
        self.combine_fitbit_data()
        
        # Align and create training data
        X, y, metadata = self.align_features_to_labels()
        
        print()
        print("=" * 60)
        print("✓ PREPARATION COMPLETE")
        print("=" * 60)
        
        return X, y, metadata
    
    def save_prepared_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metadata: Dict,
        output_dir: Union[str, Path] = "data/processed"
    ) -> Dict[str, str]:
        """
        Save prepared data to disk.
        
        Saves:
            - features.parquet: All 86 features
            - labels.parquet: CBT values
            - metadata.json: Preparation metadata
        
        Returns:
            Dict with saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}
        
        print(f"\nSaving prepared data to: {output_dir}")
        
        # Progress bar for saving
        save_steps = [
            ("features (timestamped)", lambda: X.to_parquet(output_dir / f"features_{timestamp}.parquet", index=False)),
            ("features (latest)", lambda: X.to_parquet(output_dir / "features.parquet", index=False)),
            ("labels (timestamped)", lambda: y.to_frame().to_parquet(output_dir / f"labels_{timestamp}.parquet", index=False)),
            ("labels (latest)", lambda: y.to_frame().to_parquet(output_dir / "labels.parquet", index=False)),
            ("metadata (timestamped)", lambda: self._save_json(metadata, output_dir / f"metadata_{timestamp}.json")),
            ("metadata (latest)", lambda: self._save_json(metadata, output_dir / "metadata.json")),
        ]
        
        for desc, save_func in tqdm(save_steps, desc="Saving files", disable=not HAS_TQDM):
            save_func()
        
        saved_files = {
            "features": str(output_dir / "features.parquet"),
            "labels": str(output_dir / "labels.parquet"),
            "metadata": str(output_dir / "metadata.json"),
            "features_timestamped": str(output_dir / f"features_{timestamp}.parquet"),
            "labels_timestamped": str(output_dir / f"labels_{timestamp}.parquet"),
            "metadata_timestamped": str(output_dir / f"metadata_{timestamp}.json"),
        }
        
        print()
        print("Saved files:")
        print(f"  ✓ {output_dir / 'features.parquet'} ({len(X):,} rows × {len(X.columns)} cols)")
        print(f"  ✓ {output_dir / 'labels.parquet'} ({len(y):,} values)")
        print(f"  ✓ {output_dir / 'metadata.json'}")
        
        return saved_files
    
    def _save_json(self, data: Dict, filepath: Path) -> None:
        """Helper to save JSON with proper encoding."""
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)


def create_sample_data(output_dir: str = "data/raw") -> None:
    """
    Create sample data files for testing the pipeline.
    
    Creates realistic fake data matching expected formats:
        - Fitbit heart rate (6-11 samples/min)
        - Fitbit skin temperature (1 sample/min)
        - Govee environmental (1 sample/min)
        - CBT labels (manual measurements)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating sample data files...")
    
    # Date range: 3 days
    start_date = datetime(2025, 4, 10, 0, 0, 0)
    end_date = start_date + timedelta(days=3)
    
    # ================================================
    # 1. Fitbit Heart Rate (6-11 samples per minute)
    # ================================================
    print("  Creating Fitbit heart rate data...")
    hr_records = []
    current_time = start_date
    
    for _ in tqdm(range(int((end_date - start_date).total_seconds() // 60)), desc="    HR", disable=not HAS_TQDM):
        # Variable sampling rate (6-11 per minute)
        samples_this_minute = np.random.randint(6, 12)
        intervals = sorted(np.random.choice(60, samples_this_minute, replace=False))
        
        for sec in intervals:
            ts = current_time + timedelta(seconds=int(sec))
            
            # HR varies by time of day
            hour = ts.hour
            base_hr = 70
            if 0 <= hour < 6:  # Sleep
                base_hr = 55
            elif 6 <= hour < 9:  # Waking
                base_hr = 65
            elif 12 <= hour < 14:  # After lunch
                base_hr = 75
            elif 17 <= hour < 20:  # Evening activity
                base_hr = 80
            
            hr = base_hr + np.random.randn() * 5
            hr = int(np.clip(hr, 45, 120))
            
            hr_records.append({
                "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "beats per minute": hr
            })
        
        current_time += timedelta(minutes=1)
    
    hr_df = pd.DataFrame(hr_records)
    hr_df.to_csv(output_dir / "fitbit_heart_rate.csv", index=False)
    print(f"    ✓ Created fitbit_heart_rate.csv ({len(hr_df):,} rows)")
    
    # ================================================
    # 2. Fitbit Skin Temperature (1 sample per minute)
    # ================================================
    print("  Creating Fitbit skin temperature data...")
    skin_records = []
    current_time = start_date
    
    for _ in tqdm(range(int((end_date - start_date).total_seconds() // 60)), desc="    Skin", disable=not HAS_TQDM):
        hour = current_time.hour
        
        # Skin temp deviation varies by time (higher at night)
        if 0 <= hour < 6:
            base_temp = -1.5  # Warmer during sleep
        elif 6 <= hour < 12:
            base_temp = -2.2  # Cooler in morning
        else:
            base_temp = -2.0  # Normal
        
        temp = base_temp + np.random.randn() * 0.3
        
        skin_records.append({
            "recorded_time": current_time.strftime("%Y-%m-%dT%H:%M"),
            "temperature": round(temp, 6)
        })
        
        current_time += timedelta(minutes=1)
    
    skin_df = pd.DataFrame(skin_records)
    skin_df.to_csv(output_dir / "fitbit_skin_temperature.csv", index=False)
    print(f"    ✓ Created fitbit_skin_temperature.csv ({len(skin_df):,} rows)")
    
    # ================================================
    # 3. Govee Environmental (1 sample per minute)
    # ================================================
    print("  Creating Govee environmental data...")
    env_records = []
    current_time = start_date
    
    for _ in tqdm(range(int((end_date - start_date).total_seconds() // 60)), desc="    Env", disable=not HAS_TQDM):
        hour = current_time.hour
        
        # Room temp varies slightly by time
        if 0 <= hour < 6:
            base_temp_f = 68  # Cooler at night
        elif 12 <= hour < 18:
            base_temp_f = 72  # Warmer afternoon
        else:
            base_temp_f = 70
        
        temp_f = base_temp_f + np.random.randn() * 1
        humidity = 50 + np.random.randn() * 5
        pm25 = np.random.randint(0, 10)
        
        env_records.append({
            "Timestamp for sample frequency every 1 min min": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "PM2.5(µg/m³)": pm25,
            "Temperature_Fahrenheit": round(temp_f, 2),
            "Relative_Humidity": round(humidity, 1)
        })
        
        current_time += timedelta(minutes=1)
    
    env_df = pd.DataFrame(env_records)
    env_df.to_csv(output_dir / "govee_environmental.csv", index=False)
    print(f"    ✓ Created govee_environmental.csv ({len(env_df):,} rows)")
    
    # ================================================
    # 4. CBT Labels (manual measurements, ~10 per day)
    # ================================================
    print("  Creating CBT labels...")
    cbt_records = []
    current_date = start_date.date()
    
    while current_date < end_date.date():
        # Random measurement times throughout the day
        measurement_hours = sorted(np.random.choice(range(7, 23), size=10, replace=False))
        
        for hour in measurement_hours:
            minute = np.random.randint(0, 60)
            ts = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=int(hour), minutes=int(minute))
            
            # CBT varies by time (higher in evening)
            if 6 <= hour < 10:
                base_cbt = 97.8  # Lower in morning
            elif 16 <= hour < 22:
                base_cbt = 98.8  # Higher in evening
            else:
                base_cbt = 98.3
            
            cbt = base_cbt + np.random.randn() * 0.4
            cbt = round(np.clip(cbt, 96.5, 100.5), 1)
            
            cbt_records.append({
                "Date:": ts.strftime("%m/%d/%Y"),
                "Time:": ts.strftime("%I:%M %p"),
                "CBT (Deg F):": cbt
            })
        
        current_date += timedelta(days=1)
    
    cbt_df = pd.DataFrame(cbt_records)
    cbt_df.to_csv(output_dir / "cbt_labels.csv", index=False)
    print(f"    ✓ Created cbt_labels.csv ({len(cbt_df)} measurements)")
    
    print()
    print("✓ Sample data created successfully!")
    print(f"  Location: {output_dir.absolute()}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare training data for CBT prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare data with defaults (10min window, ±15min tolerance)
  python -m src.training.prepare_data --data-dir data/raw

  # Custom window and tolerance
  python -m src.training.prepare_data --data-dir data/raw --window 15 --tolerance 20

Notes:
  - ALL 4 signals are required: heart_rate, skin_temperature, ambient_temp, humidity
  - Any sample count is valid (no minimum requirements)
  - Tolerance window slides to find nearest data
        """
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Directory containing raw data files"
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for processed output"
    )
    parser.add_argument(
        "--user-id",
        default="default_user",
        help="User identifier for the data"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DataPreparer.DEFAULT_FEATURE_WINDOW_MINUTES,
        help=f"Feature window in minutes (default: {DataPreparer.DEFAULT_FEATURE_WINDOW_MINUTES})"
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=DataPreparer.DEFAULT_ALIGNMENT_TOLERANCE_MINUTES,
        help=f"Alignment tolerance in minutes (default: {DataPreparer.DEFAULT_ALIGNMENT_TOLERANCE_MINUTES})"
    )
    
    args = parser.parse_args()
    
    preparer = DataPreparer(
        data_dir=args.data_dir,
        user_id=args.user_id,
        feature_window_minutes=args.window,
        alignment_tolerance_minutes=args.tolerance
    )
    X, y, metadata = preparer.prepare()
    preparer.save_prepared_data(X, y, metadata, args.output_dir)