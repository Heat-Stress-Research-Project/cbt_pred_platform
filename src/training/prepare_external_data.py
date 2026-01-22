"""
External Data Preparation for PROSPIE Dataset

Prepares the PROSPIE (or similar research) dataset for CBT prediction training.
This script transforms external research data to match the feature format used
by the main training pipeline.

PROSPIE Dataset Columns:
    - Environmental temperature (chamber or cooling area outside chamber) - Humidity
    - Environmental temperature (chamber or cooling area outside chamber) - Temp
    - Corerectal (CBT label - target variable)
    - Deviation_from_ParticipantBaseline (skin temperature deviation)
    - HR (heart rate)

Output:
    - Aligned features matching the 84 training features from transformations.py
    - CBT labels in Celsius
    - Metadata for tracking

IMPORTANT:
    - PROSPIE data has NO timestamps - we create synthetic timestamps
    - Each row is treated as a sequential measurement
    - We use rolling windows over row indices (not time-based)
    - This approach maintains feature compatibility with the main pipeline

Author: CBT Prediction Platform
Date: January 2026
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy import stats

# Progress bar support
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        desc = kwargs.get('desc', '')
        if desc:
            print(f"{desc}...")
        return iterable


# ============================================
# CONFIGURATION
# ============================================

class DataQuality(Enum):
    """Data quality levels for filtering."""
    HIGH = "high"       # No missing values, valid ranges
    MEDIUM = "medium"   # Some interpolated values
    LOW = "low"         # Significant interpolation


@dataclass
class PROSPIEConfig:
    """Configuration for PROSPIE data processing."""
    
    # Column mapping from PROSPIE to internal names
    column_mapping: Dict[str, str] = field(default_factory=lambda: {
        "Environmental temperature (chamber or cooling area outside chamber) - Humidity": "humidity",
        "Environmental temperature (chamber or cooling area outside chamber) - Temp": "ambient_temp",
        "Corerectal": "cbt_celsius",
        "Deviation_from_ParticipantBaseline": "skin_temperature",
        "HR": "heart_rate"
    })
    
    # Rolling window sizes (in samples, not minutes)
    # Assuming ~1 sample per minute, these correspond to 5, 20, 35 min
    rolling_windows: List[int] = field(default_factory=lambda: [5, 20, 35])
    
    # Lag interval for subsampled stats (in samples)
    lag_interval: int = 10
    
    # Minimum samples for valid feature computation
    min_samples_per_window: int = 3
    
    # Valid ranges for data cleaning
    valid_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "heart_rate": (30, 220),           # BPM
        "skin_temperature": (-10, 10),      # Deviation from baseline
        "ambient_temp": (10, 50),           # Celsius
        "humidity": (0, 100),               # Relative humidity %
        "cbt_celsius": (35, 42)             # Core body temperature
    })
    
    # Sentinel value for invalid deviation (from dataset)
    invalid_deviation_threshold: float = -9000
    
    # Synthetic timestamp settings
    sample_interval_minutes: int = 1  # Assumed interval between samples
    base_timestamp: datetime = field(default_factory=lambda: datetime(2025, 1, 1, 0, 0, 0))


class PROSPIEDataLoader:
    """
    Loads and validates PROSPIE dataset.
    
    Handles:
        - Column name normalization
        - Data type conversion
        - Invalid value detection
        - Basic statistics reporting
    """
    
    def __init__(self, config: PROSPIEConfig = None):
        self.config = config or PROSPIEConfig()
        self.raw_data: Optional[pd.DataFrame] = None
        self.load_stats: Dict = {}
    
    def load(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load PROSPIE dataset from CSV file.
        
        Args:
            filepath: Path to the PROSPIE CSV file
            
        Returns:
            DataFrame with normalized column names
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"PROSPIE data file not found: {filepath}")
        
        print(f"Loading PROSPIE data from: {filepath.name}")
        
        # Load with flexible parsing
        df = pd.read_csv(filepath)
        self.raw_data = df.copy()
        
        # Store original stats
        self.load_stats["original_rows"] = len(df)
        self.load_stats["original_columns"] = list(df.columns)
        
        # Normalize column names
        df = self._normalize_columns(df)
        
        # Convert to numeric
        df = self._convert_types(df)
        
        # Report loading stats
        self._report_load_stats(df)
        
        return df
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to internal format."""
        # Create case-insensitive mapping
        col_map = {}
        for orig_col in df.columns:
            orig_lower = orig_col.lower().strip()
            for pattern, internal_name in self.config.column_mapping.items():
                if pattern.lower() in orig_lower or orig_lower in pattern.lower():
                    col_map[orig_col] = internal_name
                    break
            
            # Direct matching for simple names
            if orig_col not in col_map:
                simple_names = {
                    "hr": "heart_rate",
                    "corerectal": "cbt_celsius",
                    "deviation_from_participantbaseline": "skin_temperature"
                }
                if orig_lower in simple_names:
                    col_map[orig_col] = simple_names[orig_lower]
        
        # Apply mapping
        df = df.rename(columns=col_map)
        
        # Verify required columns exist
        required = ["heart_rate", "skin_temperature", "ambient_temp", "humidity", "cbt_celsius"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns after normalization: {missing}")
        
        return df
    
    def _convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to appropriate numeric types."""
        numeric_cols = ["heart_rate", "skin_temperature", "ambient_temp", "humidity", "cbt_celsius"]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        return df
    
    def _report_load_stats(self, df: pd.DataFrame) -> None:
        """Report loading statistics."""
        print(f"  ✓ Loaded {len(df):,} rows")
        print(f"  Columns: {list(df.columns)}")
        
        for col in ["heart_rate", "skin_temperature", "ambient_temp", "humidity", "cbt_celsius"]:
            if col in df.columns:
                valid = df[col].notna().sum()
                pct = 100 * valid / len(df)
                print(f"    {col}: {valid:,} valid ({pct:.1f}%)")


class PROSPIEDataCleaner:
    """
    Cleans and validates PROSPIE data.
    
    Handles:
        - Invalid value removal (sentinel values like -9965.77)
        - Range validation
        - Missing value handling
        - Participant segmentation
    """
    
    def __init__(self, config: PROSPIEConfig = None):
        self.config = config or PROSPIEConfig()
        self.cleaning_stats: Dict = {}
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the PROSPIE dataset.
        
        Args:
            df: Raw PROSPIE DataFrame
            
        Returns:
            Cleaned DataFrame with invalid values handled
        """
        print("\nCleaning PROSPIE data...")
        original_len = len(df)
        df = df.copy()
        
        # Step 1: Remove rows with sentinel values in deviation
        df = self._remove_sentinel_values(df)
        
        # Step 2: Apply valid range filters
        df = self._apply_range_filters(df)
        
        # Step 3: Handle missing heart rate (interpolate or drop)
        df = self._handle_missing_hr(df)
        
        # Step 4: Identify participant segments
        df = self._identify_participants(df)
        
        # Step 5: Create synthetic timestamps
        df = self._create_timestamps(df)
        
        # Report cleaning stats
        self._report_cleaning_stats(df, original_len)
        
        return df
    
    def _remove_sentinel_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with sentinel values (e.g., -9965.77)."""
        # Identify rows with invalid deviation values
        invalid_mask = df["skin_temperature"] < self.config.invalid_deviation_threshold
        n_invalid = invalid_mask.sum()
        
        if n_invalid > 0:
            print(f"  Removing {n_invalid:,} rows with invalid deviation values")
            df = df[~invalid_mask].copy()
        
        self.cleaning_stats["sentinel_removed"] = n_invalid
        return df
    
    def _apply_range_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply valid range filters to each column."""
        total_removed = 0
        
        for col, (min_val, max_val) in self.config.valid_ranges.items():
            if col not in df.columns:
                continue
            
            # Mark out-of-range as NaN (don't drop rows yet)
            invalid_mask = (df[col] < min_val) | (df[col] > max_val)
            n_invalid = invalid_mask.sum()
            
            if n_invalid > 0:
                df.loc[invalid_mask, col] = np.nan
                total_removed += n_invalid
                print(f"  Set {n_invalid:,} out-of-range {col} values to NaN")
        
        self.cleaning_stats["range_filtered"] = total_removed
        return df
    
    def _handle_missing_hr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing heart rate values."""
        missing_hr = df["heart_rate"].isna().sum()
        
        if missing_hr > 0:
            # Interpolate short gaps (up to 5 samples)
            df["heart_rate"] = df["heart_rate"].interpolate(
                method="linear",
                limit=5,
                limit_direction="both"
            )
            
            remaining_missing = df["heart_rate"].isna().sum()
            interpolated = missing_hr - remaining_missing
            
            if interpolated > 0:
                print(f"  Interpolated {interpolated:,} missing HR values")
            
            if remaining_missing > 0:
                print(f"  {remaining_missing:,} HR values still missing (will be dropped)")
        
        self.cleaning_stats["hr_interpolated"] = missing_hr - df["heart_rate"].isna().sum()
        return df
    
    def _identify_participants(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify participant segments based on data discontinuities.
        
        Uses jumps in environmental conditions or CBT to detect new participants.
        """
        df = df.reset_index(drop=True)
        
        # Detect large jumps in ambient temp (likely new participant/session)
        temp_diff = df["ambient_temp"].diff().abs()
        cbt_diff = df["cbt_celsius"].diff().abs()
        
        # New segment if temp jumps > 3°C or CBT jumps > 1°C
        segment_breaks = (temp_diff > 3) | (cbt_diff > 1)
        segment_breaks.iloc[0] = True  # First row starts segment 1
        
        # Assign segment IDs
        df["participant_segment"] = segment_breaks.cumsum()
        
        n_segments = df["participant_segment"].nunique()
        print(f"  Identified {n_segments} participant segments")
        
        self.cleaning_stats["n_segments"] = n_segments
        return df
    
    def _create_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create synthetic timestamps for the dataset."""
        df = df.reset_index(drop=True)
        
        # Create timestamps per segment (reset for each participant)
        timestamps = []
        
        for segment_id in df["participant_segment"].unique():
            segment_mask = df["participant_segment"] == segment_id
            segment_len = segment_mask.sum()
            
            # Start each segment at a new day
            segment_start = self.config.base_timestamp + timedelta(days=int(segment_id - 1))
            
            segment_times = [
                segment_start + timedelta(minutes=i * self.config.sample_interval_minutes)
                for i in range(segment_len)
            ]
            timestamps.extend(segment_times)
        
        df["timestamp"] = pd.to_datetime(timestamps)
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        
        return df
    
    def _report_cleaning_stats(self, df: pd.DataFrame, original_len: int) -> None:
        """Report cleaning statistics."""
        final_len = len(df)
        removed = original_len - final_len
        
        print(f"\n  Cleaning Summary:")
        print(f"    Original rows: {original_len:,}")
        print(f"    Final rows: {final_len:,}")
        print(f"    Removed: {removed:,} ({100*removed/original_len:.1f}%)")
        
        # Per-column completeness
        for col in ["heart_rate", "skin_temperature", "ambient_temp", "humidity", "cbt_celsius"]:
            if col in df.columns:
                valid = df[col].notna().sum()
                pct = 100 * valid / len(df)
                print(f"    {col} completeness: {pct:.1f}%")


class PROSPIEFeatureTransformer:
    """
    Transforms PROSPIE data into ML features.
    
    Computes the same 84 training features as the main FeatureTransformer,
    but adapted for the PROSPIE dataset structure (no real timestamps).
    
    Features per signal (21 each, 4 signals = 84 total):
        - Current value (1)
        - Rolling stats: mean, median, std for 5, 20, 35 sample windows (9)
        - Lagged rolling stats: mean, median, std for 5, 20, 35 sample windows (9)
        - Simple difference (1)
        - Slope over 5 samples (1)
    """
    
    def __init__(self, config: PROSPIEConfig = None):
        self.config = config or PROSPIEConfig()
        self._feature_names: List[str] = []
    
    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Transform cleaned PROSPIE data into training features.
        
        Args:
            df: Cleaned PROSPIE DataFrame with all required columns
            
        Returns:
            X: DataFrame with 84 training features + 2 metadata
            y: Series with CBT labels (Celsius)
            metadata_df: DataFrame with sample metadata
        """
        print("\nTransforming PROSPIE data into features...")
        
        # Validate input
        self._validate_input(df)
        
        # Process each row to generate features
        features_list = []
        labels_list = []
        metadata_list = []
        skipped = {"insufficient_history": 0, "missing_values": 0}
        
        # Need minimum history for rolling calculations
        min_history = max(self.config.rolling_windows)
        
        for idx in tqdm(range(len(df)), desc="Computing features", disable=not HAS_TQDM):
            # Check if we have enough history
            if idx < min_history:
                skipped["insufficient_history"] += 1
                continue
            
            row = df.iloc[idx]
            
            # Check for missing required values in current row
            if pd.isna(row["cbt_celsius"]):
                skipped["missing_values"] += 1
                continue
            
            # Get historical window for feature computation
            history = df.iloc[max(0, idx - max(self.config.rolling_windows) - 10):idx + 1]
            
            # Compute features for this sample
            features = self._compute_sample_features(history, row, df.iloc[idx])
            
            if features is not None:
                features_list.append(features)
                labels_list.append(row["cbt_celsius"])
                metadata_list.append({
                    "timestamp": row["timestamp"],
                    "participant_segment": row["participant_segment"],
                    "row_index": idx
                })
        
        # Combine into DataFrames
        if len(features_list) == 0:
            raise ValueError("No valid samples generated. Check data quality.")
        
        X = pd.DataFrame(features_list)
        y = pd.Series(labels_list, name="cbt_celsius")
        metadata_df = pd.DataFrame(metadata_list)
        
        # Reorder columns to match expected format
        X = self._reorder_columns(X)
        self._feature_names = list(X.columns)
        
        # Report transformation stats
        self._report_transform_stats(X, y, skipped, len(df))
        
        return X, y, metadata_df
    
    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame has required columns."""
        required = ["heart_rate", "skin_temperature", "ambient_temp", "humidity", 
                   "cbt_celsius", "timestamp", "participant_segment"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def _compute_sample_features(
        self, 
        history: pd.DataFrame, 
        current_row: pd.Series,
        full_row: pd.Series
    ) -> Optional[Dict[str, float]]:
        """
        Compute all 86 features for a single sample.
        
        Args:
            history: Historical data up to and including current row
            current_row: Current row data
            full_row: Full row including metadata
            
        Returns:
            Dictionary of features or None if insufficient data
        """
        features = {}
        
        # Signal configurations: (column, prefix, current_name)
        signals = [
            ("skin_temperature", "temp", "temperature"),
            ("heart_rate", "bpm", "bpm"),
            ("ambient_temp", "temp_env", "env_Temperature_Celsius"),
            ("humidity", "humidity_env", "Relative_Humidity")
        ]
        
        for col, prefix, current_name in signals:
            signal_features = self._compute_signal_features(
                history=history,
                col=col,
                prefix=prefix,
                current_name=current_name,
                current_value=current_row[col]
            )
            
            if signal_features is None:
                return None
            
            features.update(signal_features)
        
        # Add metadata
        features["user_id"] = f"prospie_seg{int(full_row['participant_segment'])}"
        features["timestamp"] = full_row["timestamp"].isoformat() if pd.notna(full_row["timestamp"]) else ""
        
        return features
    
    def _compute_signal_features(
        self,
        history: pd.DataFrame,
        col: str,
        prefix: str,
        current_name: str,
        current_value: float
    ) -> Optional[Dict[str, float]]:
        """
        Compute all 21 features for a single signal.
        
        Features:
            - Current value (1)
            - Rolling stats: mean, median, std for 5, 20, 35 windows (9)
            - Lagged rolling stats: mean, median, std for 5, 20, 35 windows (9)
            - Simple difference (1)
            - 5-sample slope (1)
        """
        features = {}
        
        # Get valid values from history
        values = history[col].dropna().values
        
        if len(values) < self.config.min_samples_per_window:
            return None
        
        # ================================================
        # 1. Current value
        # ================================================
        if pd.isna(current_value):
            current_value = values[-1] if len(values) > 0 else np.nan
        features[current_name] = float(current_value)
        
        # ================================================
        # 2. Rolling Statistics (5, 20, 35 sample windows)
        # ================================================
        for window in self.config.rolling_windows:
            if len(values) >= window:
                window_values = values[-window:]
            else:
                window_values = values
            
            if len(window_values) > 0:
                features[f"{prefix}_mean_{window}min"] = float(np.mean(window_values))
                features[f"{prefix}_median_{window}min"] = float(np.median(window_values))
                features[f"{prefix}_std_{window}min"] = float(np.std(window_values)) if len(window_values) > 1 else 0.0
            else:
                features[f"{prefix}_mean_{window}min"] = float(current_value)
                features[f"{prefix}_median_{window}min"] = float(current_value)
                features[f"{prefix}_std_{window}min"] = 0.0
        
        # ================================================
        # 3. Lagged Rolling Statistics (subsampled)
        # ================================================
        # Subsample: take every Nth value
        lag_interval = self.config.lag_interval
        lagged_indices = list(range(0, len(values), lag_interval))
        lagged_values = values[lagged_indices] if len(lagged_indices) > 0 else values
        
        for window in self.config.rolling_windows:
            n_samples = min(window, len(lagged_values))
            
            if n_samples > 0:
                window_lagged = lagged_values[-n_samples:]
                features[f"{prefix}_mean_{window}min_lag10m"] = float(np.mean(window_lagged))
                features[f"{prefix}_median_{window}min_lag10m"] = float(np.median(window_lagged))
                features[f"{prefix}_std_{window}min_lag10m"] = float(np.std(window_lagged)) if len(window_lagged) > 1 else 0.0
            else:
                features[f"{prefix}_mean_{window}min_lag10m"] = float(current_value)
                features[f"{prefix}_median_{window}min_lag10m"] = float(current_value)
                features[f"{prefix}_std_{window}min_lag10m"] = 0.0
        
        # ================================================
        # 4. Simple Difference (current - previous)
        # ================================================
        if len(values) >= 2:
            features[f"{prefix}_diff_1"] = float(values[-1] - values[-2])
        else:
            features[f"{prefix}_diff_1"] = 0.0
        
        # ================================================
        # 5. 5-Sample Slope
        # ================================================
        window_5 = values[-5:] if len(values) >= 5 else values
        
        if len(window_5) >= 2:
            x = np.arange(len(window_5))
            slope, _, _, _, _ = stats.linregress(x, window_5)
            features[f"{prefix}_slope_5m"] = float(slope)
        else:
            features[f"{prefix}_slope_5m"] = 0.0
        
        return features
    
    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reorder columns to match expected feature order."""
        expected_order = self.get_expected_features()
        ordered_cols = [c for c in expected_order if c in df.columns]
        extra_cols = [c for c in df.columns if c not in expected_order]
        return df[ordered_cols + extra_cols]
    
    def _report_transform_stats(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        skipped: Dict, 
        total_rows: int
    ) -> None:
        """Report transformation statistics."""
        print(f"\n  Transformation Summary:")
        print(f"    Total input rows: {total_rows:,}")
        print(f"    Valid samples: {len(y):,} ({100*len(y)/total_rows:.1f}%)")
        print(f"    Features per sample: {len(X.columns)}")
        print(f"    Skipped - insufficient history: {skipped['insufficient_history']:,}")
        print(f"    Skipped - missing values: {skipped['missing_values']:,}")
        print(f"\n  CBT Statistics:")
        print(f"    Mean: {y.mean():.2f}°C")
        print(f"    Std: {y.std():.2f}°C")
        print(f"    Range: {y.min():.2f}°C - {y.max():.2f}°C")
    
    @staticmethod
    def get_expected_features() -> List[str]:
        """Return list of all 86 expected features in order."""
        features = []
        windows = [5, 20, 35]
        
        # Skin Temperature (21)
        features.append("temperature")
        for w in windows:
            features.extend([f"temp_mean_{w}min", f"temp_median_{w}min", f"temp_std_{w}min"])
        for w in windows:
            features.extend([f"temp_mean_{w}min_lag10m", f"temp_median_{w}min_lag10m", f"temp_std_{w}min_lag10m"])
        features.extend(["temp_diff_1", "temp_slope_5m"])
        
        # Heart Rate (21)
        features.append("bpm")
        for w in windows:
            features.extend([f"bpm_mean_{w}min", f"bpm_median_{w}min", f"bpm_std_{w}min"])
        for w in windows:
            features.extend([f"bpm_mean_{w}min_lag10m", f"bpm_median_{w}min_lag10m", f"bpm_std_{w}min_lag10m"])
        features.extend(["bpm_diff_1", "bpm_slope_5m"])
        
        # Ambient Temperature (21)
        features.append("env_Temperature_Celsius")
        for w in windows:
            features.extend([f"temp_env_mean_{w}min", f"temp_env_median_{w}min", f"temp_env_std_{w}min"])
        for w in windows:
            features.extend([f"temp_env_mean_{w}min_lag10m", f"temp_env_median_{w}min_lag10m", f"temp_env_std_{w}min_lag10m"])
        features.extend(["temp_env_diff_1", "temp_env_slope_5m"])
        
        # Humidity (21)
        features.append("Relative_Humidity")
        for w in windows:
            features.extend([f"humidity_env_mean_{w}min", f"humidity_env_median_{w}min", f"humidity_env_std_{w}min"])
        for w in windows:
            features.extend([f"humidity_env_mean_{w}min_lag10m", f"humidity_env_median_{w}min_lag10m", f"humidity_env_std_{w}min_lag10m"])
        features.extend(["humidity_env_diff_1", "humidity_env_slope_5m"])
        
        # Metadata (2)
        features.extend(["user_id", "timestamp"])
        
        return features
    
    @staticmethod
    def get_training_features() -> List[str]:
        """Return 84 features used for training (excludes metadata)."""
        return [f for f in PROSPIEFeatureTransformer.get_expected_features() 
                if f not in ["user_id", "timestamp"]]


class ExternalDataPreparer:
    """
    Main class for preparing external PROSPIE data for training.
    
    Orchestrates the full pipeline:
        1. Load raw data
        2. Clean and validate
        3. Transform to features
        4. Save prepared data
    
    Usage:
        preparer = ExternalDataPreparer()
        X, y, metadata = preparer.prepare("path/to/prospie.csv")
        preparer.save(X, y, metadata, "data/external_processed")
    """
    
    def __init__(self, config: PROSPIEConfig = None):
        self.config = config or PROSPIEConfig()
        self.loader = PROSPIEDataLoader(config)
        self.cleaner = PROSPIEDataCleaner(config)
        self.transformer = PROSPIEFeatureTransformer(config)
        
        # Storage
        self.raw_data: Optional[pd.DataFrame] = None
        self.cleaned_data: Optional[pd.DataFrame] = None
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None
        self.metadata: Optional[pd.DataFrame] = None
    
    def prepare(
        self, 
        filepath: Union[str, Path],
        validate_features: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """
        Prepare PROSPIE data for training.
        
        Args:
            filepath: Path to PROSPIE CSV file
            validate_features: Whether to validate feature alignment
            
        Returns:
            X: DataFrame with training features
            y: Series with CBT labels
            metadata_dict: Dictionary with preparation metadata
        """
        print("=" * 60)
        print("EXTERNAL DATA PREPARATION PIPELINE (PROSPIE)")
        print("=" * 60)
        print()
        
        # Step 1: Load raw data
        print("Step 1/3: Loading raw data...")
        self.raw_data = self.loader.load(filepath)
        
        # Step 2: Clean data
        print("\nStep 2/3: Cleaning data...")
        self.cleaned_data = self.cleaner.clean(self.raw_data)
        
        # Step 3: Transform to features
        print("\nStep 3/3: Transforming to features...")
        self.X, self.y, self.metadata = self.transformer.transform(self.cleaned_data)
        
        # Validate feature alignment if requested
        if validate_features:
            self._validate_feature_alignment()
        
        # Build metadata dictionary
        metadata_dict = self._build_metadata()
        
        print()
        print("=" * 60)
        print("✓ PREPARATION COMPLETE")
        print("=" * 60)
        
        return self.X, self.y, metadata_dict
    
    def _validate_feature_alignment(self) -> None:
        """Validate that features match expected format."""
        expected = self.transformer.get_training_features()
        actual_training = [c for c in self.X.columns if c not in ["user_id", "timestamp"]]
        
        missing = set(expected) - set(actual_training)
        extra = set(actual_training) - set(expected)
        
        if missing:
            warnings.warn(f"Missing expected features: {missing}")
        if extra:
            warnings.warn(f"Extra unexpected features: {extra}")
        
        if not missing and not extra:
            print("  ✓ Feature alignment validated")
    
    def _build_metadata(self) -> Dict:
        """Build metadata dictionary for the prepared data."""
        return {
            "source": "PROSPIE",
            "preparation_timestamp": datetime.now().isoformat(),
            "n_samples": len(self.y),
            "n_features": len(self.X.columns),
            "n_training_features": len(self.transformer.get_training_features()),
            "feature_names": list(self.X.columns),
            "training_feature_names": self.transformer.get_training_features(),
            "cbt_stats": {
                "mean": float(self.y.mean()),
                "std": float(self.y.std()),
                "min": float(self.y.min()),
                "max": float(self.y.max())
            },
            "n_participant_segments": int(self.metadata["participant_segment"].nunique()),
            "loader_stats": self.loader.load_stats,
            "cleaner_stats": self.cleaner.cleaning_stats,
            "config": {
                "rolling_windows": self.config.rolling_windows,
                "lag_interval": self.config.lag_interval,
                "sample_interval_minutes": self.config.sample_interval_minutes,
                "valid_ranges": {k: list(v) for k, v in self.config.valid_ranges.items()}
            }
        }
    
    def save(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metadata_dict: Dict,
        output_dir: Union[str, Path] = "data/external_processed"
    ) -> Dict[str, str]:
        """
        Save prepared data to disk.
        
        Saves:
            - features.parquet: All features
            - labels.parquet: CBT values
            - metadata.json: Preparation metadata
            - sample_metadata.parquet: Per-sample metadata
        
        Returns:
            Dictionary with saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\nSaving prepared data to: {output_dir}")
        
        # Save features
        features_path = output_dir / "features.parquet"
        features_ts_path = output_dir / f"features_{timestamp}.parquet"
        X.to_parquet(features_path, index=False)
        X.to_parquet(features_ts_path, index=False)
        
        # Save labels
        labels_path = output_dir / "labels.parquet"
        labels_ts_path = output_dir / f"labels_{timestamp}.parquet"
        y.to_frame().to_parquet(labels_path, index=False)
        y.to_frame().to_parquet(labels_ts_path, index=False)
        
        # Save metadata
        metadata_path = output_dir / "metadata.json"
        metadata_ts_path = output_dir / f"metadata_{timestamp}.json"
        
        with open(metadata_path, "w") as f:
            json.dump(metadata_dict, f, indent=2, default=str)
        with open(metadata_ts_path, "w") as f:
            json.dump(metadata_dict, f, indent=2, default=str)
        
        # Save sample metadata
        if self.metadata is not None:
            sample_meta_path = output_dir / "sample_metadata.parquet"
            self.metadata.to_parquet(sample_meta_path, index=False)
        
        saved_files = {
            "features": str(features_path),
            "labels": str(labels_path),
            "metadata": str(metadata_path),
            "features_timestamped": str(features_ts_path),
            "labels_timestamped": str(labels_ts_path),
            "metadata_timestamped": str(metadata_ts_path)
        }
        
        print(f"  ✓ {features_path} ({len(X):,} rows × {len(X.columns)} cols)")
        print(f"  ✓ {labels_path} ({len(y):,} values)")
        print(f"  ✓ {metadata_path}")
        
        return saved_files


def prepare_prospie_for_training(
    input_path: Union[str, Path],
    output_dir: Union[str, Path] = "data/external_processed",
    config: PROSPIEConfig = None
) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Convenience function to prepare PROSPIE data.
    
    Args:
        input_path: Path to PROSPIE CSV file
        output_dir: Directory for output files
        config: Optional configuration override
        
    Returns:
        X, y, metadata tuple
    """
    preparer = ExternalDataPreparer(config)
    X, y, metadata = preparer.prepare(input_path)
    preparer.save(X, y, metadata, output_dir)
    return X, y, metadata


# ============================================
# CLI INTERFACE
# ============================================

def main():
    """Command-line interface for PROSPIE data preparation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare PROSPIE external dataset for CBT prediction training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m src.training.prepare_external_data --input data/Cleaned_PROSPIE_Dataset.csv

  # Custom output directory
  python -m src.training.prepare_external_data --input data/prospie.csv --output data/prospie_processed

  # With custom rolling windows
  python -m src.training.prepare_external_data --input data/prospie.csv --windows 5 15 30

Column Mapping:
  PROSPIE Column                                              -> Internal Name
  -------------------------------------------------------------------------
  Environmental temperature... - Humidity                      -> humidity
  Environmental temperature... - Temp                          -> ambient_temp
  Corerectal                                                   -> cbt_celsius
  Deviation_from_ParticipantBaseline                          -> skin_temperature
  HR                                                           -> heart_rate

Notes:
  - Invalid deviation values (< -9000) are automatically removed
  - Missing HR values are interpolated (up to 5 samples)
  - Participant segments are automatically detected
  - Synthetic timestamps are created for feature computation
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to PROSPIE CSV file"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/external_processed",
        help="Output directory for processed data (default: data/external_processed)"
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[5, 20, 35],
        help="Rolling window sizes in samples (default: 5 20 35)"
    )
    parser.add_argument(
        "--lag-interval",
        type=int,
        default=10,
        help="Lag interval for subsampled stats (default: 10)"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip feature alignment validation"
    )
    
    args = parser.parse_args()
    
    # Build config
    config = PROSPIEConfig()
    config.rolling_windows = args.windows
    config.lag_interval = args.lag_interval
    
    # Run preparation
    preparer = ExternalDataPreparer(config)
    X, y, metadata = preparer.prepare(
        args.input,
        validate_features=not args.no_validate
    )
    preparer.save(X, y, metadata, args.output)
    
    return X, y, metadata


if __name__ == "__main__":
    main()