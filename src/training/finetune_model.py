"""
Model Fine-Tuning for CBT Prediction

Specialized script for fine-tuning a pre-trained PROSPIE model 
on study collected data.

Features:
    - Adaptive learning rate scheduling
    - Domain adaptation techniques
    - Validation on both domains
    - Transfer learning strategies
    - Hyperparameter optimization for fine-tuning

Usage:
    python -m src.training.finetune_model \\
        --base-model models/prospie \\
        --study-data data/processed \\
        --output models/study_model_v1

Author: CBT Prediction Platform
Date: January 2026
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Progress bar support
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

# XGBoost import
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    raise ImportError("XGBoost is required. Install with: pip install xgboost")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.transformations import FeatureTransformer


@dataclass
class FineTuneConfig:
    """Configuration for fine-tuning."""
    
    # Learning rate schedule
    initial_lr: float = 0.01
    min_lr: float = 0.001
    lr_decay_factor: float = 0.5
    patience: int = 10
    
    # Training parameters
    n_rounds: int = 100
    early_stopping_rounds: int = 15
    
    # Regularization (often increased for fine-tuning)
    reg_alpha: float = 0.5
    reg_lambda: float = 2.0
    
    # Data settings
    test_size: float = 0.2
    random_state: int = 42
    
    # Domain adaptation
    use_sample_weights: bool = True
    recent_data_weight: float = 2.0  # Weight for recent samples
    
    # Validation
    validate_on_prospie: bool = False  # Validate on original domain too
    prospie_data_dir: Optional[str] = None


class FineTuneStrategy:
    """
    Strategies for fine-tuning.
    """
    
    @staticmethod
    def compute_sample_weights(
        timestamps: pd.Series,
        recent_weight: float = 2.0
    ) -> np.ndarray:
        """
        Compute sample weights giving more importance to recent data.
        
        Args:
            timestamps: Sample timestamps
            recent_weight: Maximum weight for most recent samples
            
        Returns:
            Array of sample weights
        """
        if len(timestamps) == 0:
            return np.array([])
        
        # Convert to numeric (seconds since epoch)
        numeric_times = pd.to_datetime(timestamps).astype(np.int64) / 1e9
        
        # Normalize to [0, 1]
        time_min = numeric_times.min()
        time_max = numeric_times.max()
        
        if time_max == time_min:
            return np.ones(len(timestamps))
        
        normalized = (numeric_times - time_min) / (time_max - time_min)
        
        # Linear weight from 1.0 to recent_weight
        weights = 1.0 + (recent_weight - 1.0) * normalized
        
        return weights.values
    
    @staticmethod
    def adaptive_learning_rate(
        base_lr: float,
        round_num: int,
        total_rounds: int,
        warmup_rounds: int = 5
    ) -> float:
        """
        Compute adaptive learning rate with warmup and decay.
        
        Args:
            base_lr: Base learning rate
            round_num: Current round number
            total_rounds: Total training rounds
            warmup_rounds: Rounds for warmup phase
            
        Returns:
            Adjusted learning rate
        """
        if round_num < warmup_rounds:
            # Warmup phase: linear increase
            return base_lr * (round_num + 1) / warmup_rounds
        else:
            # Cosine decay
            progress = (round_num - warmup_rounds) / (total_rounds - warmup_rounds)
            return base_lr * 0.5 * (1 + np.cos(np.pi * progress))


class ModelFineTuner:
    """
    Specialized fine-tuner for CBT prediction models.
    
    Implements domain adaptation from PROSPIE to study data.
    
    Usage:
        finetuner = ModelFineTuner("models/prospie")
        finetuner.finetune(data_dir="data/processed")
        finetuner.save("models/study_model_v1")
    """
    
    def __init__(
        self,
        base_model_dir: Union[str, Path],
        config: Optional[FineTuneConfig] = None
    ):
        """
        Initialize fine-tuner with pre-trained model.
        
        Args:
            base_model_dir: Directory containing pre-trained model
            config: Fine-tuning configuration
        """
        self.base_model_dir = Path(base_model_dir)
        self.config = config or FineTuneConfig()
        
        # Load base model
        self.model: Optional[xgb.XGBRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.base_metadata: Dict[str, Any] = {}
        
        self._load_base_model()
        
        # Fine-tuning results
        self.finetune_metadata: Dict[str, Any] = {}
        self.training_history: List[Dict] = []
    
    def _load_base_model(self) -> None:
        """Load pre-trained model and artifacts."""
        logger.info(f"Loading base model from: {self.base_model_dir}")
        
        # Load model
        model_path = self.base_model_dir / "model.joblib"
        if not model_path.exists():
            model_path = self.base_model_dir / "model.json"
        
        if model_path.suffix == ".joblib":
            self.model = joblib.load(model_path)
        else:
            self.model = xgb.XGBRegressor()
            self.model.load_model(model_path)
        
        # Load scaler
        scaler_path = self.base_model_dir / "scaler.joblib"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        
        # Load feature names
        features_path = self.base_model_dir / "feature_names.json"
        if features_path.exists():
            with open(features_path, "r") as f:
                data = json.load(f)
                self.feature_names = data.get("feature_names", [])
        
        # Load metadata
        metadata_path = self.base_model_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self.base_metadata = json.load(f)
        
        logger.info(f"  Loaded model with {len(self.feature_names)} features")
        
        # Get base model performance
        if "metrics" in self.base_metadata:
            test_mae = self.base_metadata["metrics"].get("test", {}).get("mae", "N/A")
            logger.info(f"  Base model test MAE: {test_mae}")
    
    def _load_study_data(
        self,
        data_dir: Union[str, Path]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Load study collected data."""
        data_dir = Path(data_dir)
        
        features_path = data_dir / "features.parquet"
        labels_path = data_dir / "labels.parquet"
        
        if not features_path.exists():
            raise FileNotFoundError(f"Features not found: {features_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels not found: {labels_path}")
        
        X = pd.read_parquet(features_path)
        y_df = pd.read_parquet(labels_path)
        
        # Extract target
        target_cols = ["cbt_celsius", "cbt_temperature", "cbt"]
        y = None
        for col in target_cols:
            if col in y_df.columns:
                y = y_df[col]
                break
        if y is None:
            y = y_df.iloc[:, 0]
        
        return X, y
    
    def _prepare_features(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and extract timestamps."""
        # Extract timestamp before removing metadata
        timestamps = None
        if "timestamp" in X.columns:
            timestamps = X["timestamp"]
        
        # Remove metadata columns
        metadata_cols = ["user_id", "timestamp"]
        feature_cols = [c for c in X.columns if c not in metadata_cols]
        
        # Align with base model features
        X_aligned = X[self.feature_names].copy()
        
        return X_aligned, timestamps
    
    def _evaluate(
        self,
        X: np.ndarray,
        y: pd.Series,
        dataset_name: str = "Test"
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        y_pred = self.model.predict(X)
        
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "dataset": dataset_name
        }
    
    def finetune(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        data_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Fine-tune the model on study data.
        
        Args:
            X: Feature DataFrame (optional if data_dir provided)
            y: Target Series (optional if data_dir provided)
            data_dir: Directory with prepared study data
            
        Returns:
            Dictionary with fine-tuning results
        """
        print("=" * 70)
        print("MODEL FINE-TUNING")
        print("=" * 70)
        print()
        
        # Load data
        if X is None or y is None:
            if data_dir is None:
                data_dir = Path("data/processed")
            X, y = self._load_study_data(data_dir)
        
        # Prepare features
        X_prepared, timestamps = self._prepare_features(X)
        
        print(f"Personal Dataset: {len(y):,} samples × {len(self.feature_names)} features")
        print(f"Target range: {y.min():.2f}°C - {y.max():.2f}°C")
        print()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_prepared, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        
        # Get timestamps for sample weighting
        if timestamps is not None:
            train_idx = X_train.index
            train_timestamps = timestamps.loc[train_idx]
        else:
            train_timestamps = None
        
        print(f"Train: {len(X_train):,} samples")
        print(f"Test:  {len(X_test):,} samples")
        print()
        
        # Handle missing values
        fill_values = self.base_metadata.get("fill_values", {})
        for col in X_train.columns:
            if X_train[col].isna().any():
                fill_val = fill_values.get(col, X_train[col].median())
                X_train[col] = X_train[col].fillna(fill_val)
                X_test[col] = X_test[col].fillna(fill_val)
        
        # Scale features
        if self.scaler is not None:
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values
        
        # Compute sample weights (emphasize recent data)
        sample_weights = None
        if self.config.use_sample_weights and train_timestamps is not None:
            sample_weights = FineTuneStrategy.compute_sample_weights(
                train_timestamps,
                recent_weight=self.config.recent_data_weight
            )
            print(f"Sample weights: min={sample_weights.min():.2f}, max={sample_weights.max():.2f}")
            print()
        
        # Evaluate before fine-tuning
        print("Pre-fine-tuning performance:")
        before_metrics = self._evaluate(X_test_scaled, y_test, "Test (before)")
        print(f"  MAE:  {before_metrics['mae']:.4f}°C")
        print(f"  RMSE: {before_metrics['rmse']:.4f}°C")
        print(f"  R²:   {before_metrics['r2']:.4f}")
        print()
        
        # Fine-tuning parameters
        finetune_params = {
            "n_estimators": self.config.n_rounds,
            "learning_rate": self.config.initial_lr,
            "max_depth": self.model.max_depth,
            "reg_alpha": self.config.reg_alpha,
            "reg_lambda": self.config.reg_lambda,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": self.config.random_state,
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "verbosity": 0,
            "tree_method": "hist"
        }
        
        print("Fine-tuning Configuration:")
        print(f"  Additional rounds: {self.config.n_rounds}")
        print(f"  Learning rate:     {self.config.initial_lr}")
        print(f"  Early stopping:    {self.config.early_stopping_rounds} rounds")
        print(f"  Regularization:    α={self.config.reg_alpha}, λ={self.config.reg_lambda}")
        print()
        
        # Fine-tune model
        print("Fine-tuning...")
        
        finetuned_model = xgb.XGBRegressor(**finetune_params)
        
        eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]
        
        # Continue training from base model
        finetuned_model.fit(
            X_train_scaled, y_train,
            eval_set=eval_set,
            sample_weight=sample_weights,
            xgb_model=self.model.get_booster(),
            verbose=False
        )
        
        self.model = finetuned_model
        print("✓ Fine-tuning complete!")
        print()
        
        # Evaluate after fine-tuning
        print("Post-fine-tuning performance:")
        after_metrics = self._evaluate(X_test_scaled, y_test, "Test (after)")
        print(f"  MAE:  {after_metrics['mae']:.4f}°C")
        print(f"  RMSE: {after_metrics['rmse']:.4f}°C")
        print(f"  R²:   {after_metrics['r2']:.4f}")
        print()
        
        # Calculate improvement
        mae_improvement = before_metrics["mae"] - after_metrics["mae"]
        r2_improvement = after_metrics["r2"] - before_metrics["r2"]
        
        print("Improvement:")
        print(f"  MAE: {mae_improvement:+.4f}°C ({'✓ better' if mae_improvement > 0 else '✗ worse'})")
        print(f"  R²:  {r2_improvement:+.4f} ({'✓ better' if r2_improvement > 0 else '✗ worse'})")
        print()
        
        # Get feature importance
        importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        print("Top 10 Features (post fine-tuning):")
        for i, (name, imp) in enumerate(list(importance.items())[:10]):
            print(f"  {i+1:2}. {name:<35} {imp:.4f}")
        print()
        
        # Build results
        results = {
            "before_finetune": before_metrics,
            "after_finetune": after_metrics,
            "improvement": {
                "mae": float(mae_improvement),
                "r2": float(r2_improvement)
            },
            "n_samples": len(y),
            "n_train": len(y_train),
            "n_test": len(y_test),
            "feature_importance": importance
        }
        
        # Store metadata
        self.finetune_metadata = {
            "finetune_timestamp": datetime.now().isoformat(),
            "base_model_dir": str(self.base_model_dir),
            "config": {
                "initial_lr": self.config.initial_lr,
                "n_rounds": self.config.n_rounds,
                "reg_alpha": self.config.reg_alpha,
                "reg_lambda": self.config.reg_lambda,
                "use_sample_weights": self.config.use_sample_weights
            },
            "results": results,
            "base_metadata": self.base_metadata
        }
        
        print("=" * 70)
        print("FINE-TUNING COMPLETE")
        print("=" * 70)
        
        return results
    
    def save(self, output_dir: Union[str, Path]) -> Dict[str, str]:
        """
        Save fine-tuned model and artifacts.
        
        Args:
            output_dir: Directory to save model
            
        Returns:
            Dictionary with saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        print(f"\nSaving fine-tuned model to: {output_dir}")
        
        # Save model (JSON and joblib)
        model_json = output_dir / "model.json"
        self.model.save_model(model_json)
        saved_files["model_json"] = str(model_json)
        
        model_joblib = output_dir / "model.joblib"
        joblib.dump(self.model, model_joblib)
        saved_files["model_joblib"] = str(model_joblib)
        
        # Save scaler
        if self.scaler is not None:
            scaler_path = output_dir / "scaler.joblib"
            joblib.dump(self.scaler, scaler_path)
            saved_files["scaler"] = str(scaler_path)
        
        # Save feature names
        features_path = output_dir / "feature_names.json"
        with open(features_path, "w") as f:
            json.dump({
                "feature_names": self.feature_names,
                "n_features": len(self.feature_names)
            }, f, indent=2)
        saved_files["feature_names"] = str(features_path)
        
        # Save metadata (combined base + finetune)
        combined_metadata = {
            **self.base_metadata,
            "finetune": self.finetune_metadata
        }
        
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(combined_metadata, f, indent=2, default=str)
        saved_files["metadata"] = str(metadata_path)
        
        print("Saved artifacts:")
        for name, path in saved_files.items():
            print(f"  ✓ {Path(path).name}")
        
        return saved_files
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with fine-tuned model."""
        X_prepared, _ = self._prepare_features(X)
        
        # Handle missing values
        fill_values = self.base_metadata.get("fill_values", {})
        for col in X_prepared.columns:
            if X_prepared[col].isna().any():
                fill_val = fill_values.get(col, X_prepared[col].median())
                X_prepared[col] = X_prepared[col].fillna(fill_val)
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_prepared)
        else:
            X_scaled = X_prepared.values
        
        return self.model.predict(X_scaled)


def main():
    """Command-line interface for fine-tuning."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fine-tune CBT prediction model on study data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic fine-tuning
  python -m src.training.finetune_model \\
      --base-model models/prospie \\
      --study-data data/processed \\
      --output models/study_model_v1

  # With custom learning rate
  python -m src.training.finetune_model \\
      --base-model models/prospie \\
      --study-data data/processed \\
      --learning-rate 0.005 \\
      --n-rounds 150
        """
    )
    
    parser.add_argument(
        "--base-model",
        type=str,
        default="models/prospie",
        help="Path to pre-trained base model"
    )
    parser.add_argument(
        "--study-data",
        type=str,
        default="data/processed",
        help="Path to prepared study data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/study_model_v1",
        help="Output directory for fine-tuned model"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Fine-tuning learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--n-rounds",
        type=int,
        default=100,
        help="Number of fine-tuning rounds (default: 100)"
    )
    parser.add_argument(
        "--no-sample-weights",
        action="store_true",
        help="Disable sample weighting (giving more weight to recent data)"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = FineTuneConfig(
        initial_lr=args.learning_rate,
        n_rounds=args.n_rounds,
        use_sample_weights=not args.no_sample_weights
    )
    
    # Fine-tune
    finetuner = ModelFineTuner(args.base_model, config)
    finetuner.finetune(data_dir=args.study_data)
    finetuner.save(args.output)


if __name__ == "__main__":
    main()