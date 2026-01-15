"""

Model Training for CBT Prediction

Trains an XGBoost model to predict Core Body Temperature from 
wearable (Fitbit) and environmental (Govee) features.

Usage:
    python -m src.training.train_model --data-dir data/processed --output-dir models/
    
    Or use ModelTrainer class directly:
        trainer = ModelTrainer()
        trainer.train(X, y)
        trainer.save("models/")

Features:
    - 84 training features (excludes user_id, timestamp metadata)
    - XGBoost regressor with optional sklearn fallback
    - Feature scaling with StandardScaler
    - Cross-validation with progress bars
    - Feature importance analysis
    - Model persistence with joblib
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List, Union

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        """Fallback for tqdm if not installed."""
        return iterable

# Try to import XGBoost, fall back to sklearn if not available
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    from sklearn.ensemble import GradientBoostingRegressor

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.transformations import FeatureTransformer


class ModelTrainer:
    """
    Trains and saves CBT prediction model.
    
    Usage:
        trainer = ModelTrainer()
        metrics = trainer.train(X, y)
        trainer.save("models/")
    
    Features Used:
        84 training features from FeatureTransformer.get_training_features()
        Excludes: user_id, timestamp (metadata only)
    """
    
    def __init__(
        self,
        model_type: str = "xgboost",
        test_size: float = 0.2,
        random_state: int = 42,
        scale_features: bool = True,
        n_cv_folds: int = 5
    ):
        """
        Initialize ModelTrainer.
        
        Args:
            model_type: Type of model ("xgboost" or "sklearn")
            test_size: Fraction of data for testing (default: 0.2)
            random_state: Random seed for reproducibility
            scale_features: Whether to standardize features (default: True)
            n_cv_folds: Number of cross-validation folds (default: 5)
        """
        self.model_type = model_type
        self.test_size = test_size
        self.random_state = random_state
        self.scale_features = scale_features
        self.n_cv_folds = n_cv_folds
        
        self.model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.training_metadata: Dict = {}
        
        # Get expected training features
        self.expected_features = FeatureTransformer.get_training_features()
        
    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for training by selecting only training columns.
        
        Removes metadata columns (user_id, timestamp) if present.
        
        Args:
            X: Input DataFrame with all features
        
        Returns:
            DataFrame with only training features (84 columns)
        """
        # Get columns to use (exclude metadata)
        metadata_cols = ["user_id", "timestamp"]
        training_cols = [c for c in X.columns if c not in metadata_cols]
        
        # Validate features
        missing = [f for f in self.expected_features if f not in training_cols]
        if missing:
            print(f"Warning: Missing expected features: {missing[:5]}...")
        
        extra = [f for f in training_cols if f not in self.expected_features]
        if extra:
            print(f"Warning: Extra features (will be dropped): {extra[:5]}...")
            training_cols = [c for c in training_cols if c in self.expected_features]
        
        return X[training_cols]
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        hyperparams: Dict = None,
        verbose: bool = True
    ) -> Dict:
        """
        Train the model with progress bars.
        
        Args:
            X: Feature DataFrame (86 columns, will filter to 84 training)
            y: Target Series (CBT values in Celsius)
            hyperparams: Optional hyperparameters for the model
            verbose: Whether to print progress information
        
        Returns:
            Dict with training metrics
        """
        print("=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)
        print()
        
        # Prepare features (remove metadata columns)
        print("Preparing features...")
        X_train_full = self._prepare_features(X)
        self.feature_names = list(X_train_full.columns)
        
        print(f"  Input features: {len(X.columns)}")
        print(f"  Training features: {len(self.feature_names)}")
        print(f"  Samples: {len(y)}")
        print()
        
        # Split data
        print(f"Splitting data (test_size={self.test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_full, y,
            test_size=self.test_size,
            random_state=self.random_state
        )
        print(f"  Train: {len(X_train)} samples")
        print(f"  Test: {len(X_test)} samples")
        print()
        
        # Handle NaN values
        nan_count = X_train.isna().sum().sum()
        if nan_count > 0:
            print(f"Handling {nan_count} NaN values...")
            # Fill with column medians from training set
            for col in tqdm(X_train.columns, desc="  Filling NaN", disable=not HAS_TQDM):
                if X_train[col].isna().any():
                    median_val = X_train[col].median()
                    X_train[col] = X_train[col].fillna(median_val)
                    X_test[col] = X_test[col].fillna(median_val)
            print()
        
        # Scale features
        if self.scale_features:
            print("Scaling features...")
            self.scaler = StandardScaler()
            
            with tqdm(total=2, desc="  Fitting scaler", disable=not HAS_TQDM) as pbar:
                X_train_scaled = self.scaler.fit_transform(X_train)
                pbar.update(1)
                X_test_scaled = self.scaler.transform(X_test)
                pbar.update(1)
            print()
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values
        
        # Default hyperparameters
        if hyperparams is None:
            hyperparams = {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "min_child_weight": 3,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.random_state
            }
        
        # Create model
        print(f"Creating {self.model_type} model...")
        print(f"  Hyperparameters:")
        for k, v in hyperparams.items():
            print(f"    {k}: {v}")
        print()
        
        if self.model_type == "xgboost" and HAS_XGBOOST:
            # XGBoost with progress callback
            self.model = xgb.XGBRegressor(
                **hyperparams,
                verbosity=0
            )
        else:
            if self.model_type == "xgboost":
                print("  ⚠ XGBoost not installed, using sklearn GradientBoosting")
            # Remove XGBoost-specific params for sklearn
            sklearn_params = {
                k: v for k, v in hyperparams.items() 
                if k in ["n_estimators", "max_depth", "learning_rate", "random_state", "subsample"]
            }
            self.model = GradientBoostingRegressor(**sklearn_params)
        
        # Train model with progress
        print("Training model...")
        
        if self.model_type == "xgboost" and HAS_XGBOOST:
            # XGBoost training with eval set for progress
            eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]
            
            # Custom callback for progress bar
            n_estimators = hyperparams.get("n_estimators", 100)
            
            with tqdm(total=n_estimators, desc="  Training", disable=not HAS_TQDM) as pbar:
                self.model.fit(
                    X_train_scaled, y_train,
                    eval_set=eval_set,
                    verbose=False
                )
                pbar.update(n_estimators)
        else:
            # sklearn training (no built-in progress)
            with tqdm(total=1, desc="  Training", disable=not HAS_TQDM) as pbar:
                self.model.fit(X_train_scaled, y_train)
                pbar.update(1)
        
        print("  ✓ Training complete!")
        print()
        
        # Evaluate
        print("Evaluating model...")
        with tqdm(total=2, desc="  Predicting", disable=not HAS_TQDM) as pbar:
            train_pred = self.model.predict(X_train_scaled)
            pbar.update(1)
            test_pred = self.model.predict(X_test_scaled)
            pbar.update(1)
        
        metrics = {
            "train": {
                "mae": float(mean_absolute_error(y_train, train_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_train, train_pred))),
                "r2": float(r2_score(y_train, train_pred)),
                "n_samples": len(y_train)
            },
            "test": {
                "mae": float(mean_absolute_error(y_test, test_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
                "r2": float(r2_score(y_test, test_pred)),
                "n_samples": len(y_test)
            }
        }
        
        print()
        print("  Results:")
        print(f"  ┌{'─'*50}┐")
        print(f"  │ {'TRAIN':<10} MAE: {metrics['train']['mae']:.4f}°C  "
              f"RMSE: {metrics['train']['rmse']:.4f}°C  "
              f"R²: {metrics['train']['r2']:.4f} │")
        print(f"  │ {'TEST':<10} MAE: {metrics['test']['mae']:.4f}°C  "
              f"RMSE: {metrics['test']['rmse']:.4f}°C  "
              f"R²: {metrics['test']['r2']:.4f} │")
        print(f"  └{'─'*50}┘")
        print()
        
        # Cross-validation with progress
        print(f"Running {self.n_cv_folds}-fold cross-validation...")
        cv_scores = self._cross_validate(X_train_scaled, y_train)
        
        metrics["cv"] = {
            "mae_mean": float(np.mean(cv_scores["mae"])),
            "mae_std": float(np.std(cv_scores["mae"])),
            "rmse_mean": float(np.mean(cv_scores["rmse"])),
            "rmse_std": float(np.std(cv_scores["rmse"])),
            "r2_mean": float(np.mean(cv_scores["r2"])),
            "r2_std": float(np.std(cv_scores["r2"])),
            "fold_scores": cv_scores
        }
        
        print(f"  CV MAE:  {metrics['cv']['mae_mean']:.4f}°C (±{metrics['cv']['mae_std']:.4f})")
        print(f"  CV RMSE: {metrics['cv']['rmse_mean']:.4f}°C (±{metrics['cv']['rmse_std']:.4f})")
        print(f"  CV R²:   {metrics['cv']['r2_mean']:.4f} (±{metrics['cv']['r2_std']:.4f})")
        print()
        
        # Feature importance
        print("Analyzing feature importance...")
        importance = self._get_feature_importance()
        
        print("  Top 15 most important features:")
        print(f"  ┌{'─'*45}┐")
        for i, (name, imp) in enumerate(importance[:15]):
            bar = "█" * int(imp * 50)
            print(f"  │ {i+1:2}. {name:<30} {imp:.4f} │")
        print(f"  └{'─'*45}┘")
        print()
        
        metrics["feature_importance"] = {name: float(imp) for name, imp in importance}
        
        # Store metadata
        self.training_metadata = {
            "model_type": self.model_type,
            "model_class": self.model.__class__.__name__,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "hyperparameters": hyperparams,
            "scale_features": self.scale_features,
            "n_cv_folds": self.n_cv_folds,
            "trained_at": datetime.now().isoformat(),
            "metrics": metrics,
            "target_stats": {
                "mean": float(y.mean()),
                "std": float(y.std()),
                "min": float(y.min()),
                "max": float(y.max())
            }
        }
        
        print("=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        
        return metrics
    
    def _cross_validate(
        self, 
        X: np.ndarray, 
        y: pd.Series
    ) -> Dict[str, List[float]]:
        """
        Perform cross-validation with progress bar.
        
        Args:
            X: Scaled feature array
            y: Target values
        
        Returns:
            Dict with lists of scores for each fold
        """
        kf = KFold(n_splits=self.n_cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_scores = {"mae": [], "rmse": [], "r2": []}
        
        for fold, (train_idx, val_idx) in enumerate(
            tqdm(kf.split(X), total=self.n_cv_folds, desc="  CV Folds", disable=not HAS_TQDM)
        ):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Clone and train model
            if self.model_type == "xgboost" and HAS_XGBOOST:
                fold_model = xgb.XGBRegressor(**self.model.get_params())
            else:
                fold_model = GradientBoostingRegressor(**self.model.get_params())
            
            fold_model.fit(X_fold_train, y_fold_train)
            fold_pred = fold_model.predict(X_fold_val)
            
            cv_scores["mae"].append(mean_absolute_error(y_fold_val, fold_pred))
            cv_scores["rmse"].append(np.sqrt(mean_squared_error(y_fold_val, fold_pred)))
            cv_scores["r2"].append(r2_score(y_fold_val, fold_pred))
        
        return cv_scores
    
    def _get_feature_importance(self) -> List[Tuple[str, float]]:
        """Get feature importance from trained model, sorted descending."""
        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        else:
            importance = np.zeros(len(self.feature_names))
        
        pairs = list(zip(self.feature_names, importance))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature DataFrame (can include metadata columns)
        
        Returns:
            Array of predicted CBT values (Celsius)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Select only training features
        X_pred = self._prepare_features(X)
        
        # Ensure correct column order
        X_pred = X_pred[self.feature_names]
        
        # Handle NaN
        X_pred = X_pred.fillna(X_pred.median())
        
        if self.scale_features and self.scaler is not None:
            X_scaled = self.scaler.transform(X_pred)
        else:
            X_scaled = X_pred.values
        
        return self.model.predict(X_scaled)
    
    def save(self, output_dir: Union[str, Path] = "models/") -> Dict[str, str]:
        """
        Save model and all artifacts.
        
        Saves:
            - model.joblib: Trained model
            - scaler.joblib: Feature scaler (if used)
            - model_metadata.json: Training metadata and metrics
            - features.json: Feature names list
        
        Args:
            output_dir: Directory to save model artifacts
        
        Returns:
            Dict with paths to saved files
        """
        if self.model is None:
            raise ValueError("No model to save. Call train() first.")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        print("Saving model artifacts...")
        
        with tqdm(total=4, desc="  Saving", disable=not HAS_TQDM) as pbar:
            # Save model
            model_path = output_dir / "model.joblib"
            joblib.dump(self.model, model_path)
            saved_files["model"] = str(model_path)
            pbar.update(1)
            
            # Save scaler (if used)
            if self.scaler is not None:
                scaler_path = output_dir / "scaler.joblib"
                joblib.dump(self.scaler, scaler_path)
                saved_files["scaler"] = str(scaler_path)
            pbar.update(1)
            
            # Save metadata
            metadata_path = output_dir / "model_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(self.training_metadata, f, indent=2, default=str)
            saved_files["metadata"] = str(metadata_path)
            pbar.update(1)
            
            # Save feature names
            features_path = output_dir / "features.json"
            with open(features_path, "w") as f:
                json.dump({
                    "feature_names": self.feature_names,
                    "n_features": len(self.feature_names),
                    "expected_features": self.expected_features
                }, f, indent=2)
            saved_files["features"] = str(features_path)
            pbar.update(1)
        
        print()
        print("Saved files:")
        for name, path in saved_files.items():
            print(f"  ✓ {name}: {path}")
        
        return saved_files
    
    @classmethod
    def load(cls, model_dir: Union[str, Path]) -> "ModelTrainer":
        """
        Load a trained model from disk.
        
        Args:
            model_dir: Directory containing saved model artifacts
        
        Returns:
            ModelTrainer instance with loaded model
        """
        model_dir = Path(model_dir)
        
        print(f"Loading model from {model_dir}...")
        
        trainer = cls()
        
        with tqdm(total=4, desc="  Loading", disable=not HAS_TQDM) as pbar:
            # Load model
            trainer.model = joblib.load(model_dir / "model.joblib")
            pbar.update(1)
            
            # Load scaler if exists
            scaler_path = model_dir / "scaler.joblib"
            if scaler_path.exists():
                trainer.scaler = joblib.load(scaler_path)
                trainer.scale_features = True
            pbar.update(1)
            
            # Load metadata
            metadata_path = model_dir / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    trainer.training_metadata = json.load(f)
                trainer.feature_names = trainer.training_metadata.get("feature_names", [])
                trainer.model_type = trainer.training_metadata.get("model_type", "unknown")
            pbar.update(1)
            
            # Load feature names
            features_path = model_dir / "features.json"
            if features_path.exists():
                with open(features_path, "r") as f:
                    data = json.load(f)
                trainer.feature_names = data.get("feature_names", trainer.feature_names)
            pbar.update(1)
        
        print()
        print(f"  ✓ Model type: {trainer.model_type}")
        print(f"  ✓ Features: {len(trainer.feature_names)}")
        if trainer.training_metadata.get("metrics"):
            test_mae = trainer.training_metadata["metrics"]["test"]["mae"]
            print(f"  ✓ Test MAE: {test_mae:.4f}°C")
        
        return trainer


def main():
    """Command-line interface for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train CBT prediction model")
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Directory containing prepared data (features.parquet, labels.parquet)"
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory to save trained model"
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Don't scale features"
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of boosting rounds"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Maximum tree depth"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Load prepared data
    print("Loading prepared data...")
    print(f"  Data directory: {data_dir}")
    
    features_path = data_dir / "features.parquet"
    labels_path = data_dir / "labels.parquet"
    
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    X = pd.read_parquet(features_path)
    y_df = pd.read_parquet(labels_path)
    
    # Handle different possible column names for target
    if "cbt_celsius" in y_df.columns:
        y = y_df["cbt_celsius"]
    elif "cbt_temperature" in y_df.columns:
        y = y_df["cbt_temperature"]
    else:
        y = y_df.iloc[:, 0]  # First column
    
    print(f"  Features shape: {X.shape}")
    print(f"  Labels shape: {y.shape}")
    print(f"  Target range: {y.min():.2f}°C - {y.max():.2f}°C")
    print()
    
    # Custom hyperparameters
    hyperparams = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    }
    
    # Train model
    trainer = ModelTrainer(
        scale_features=not args.no_scale,
        n_cv_folds=args.cv_folds
    )
    trainer.train(X, y, hyperparams=hyperparams)
    
    # Save model
    print()
    trainer.save(args.output_dir)


if __name__ == "__main__":
    main()