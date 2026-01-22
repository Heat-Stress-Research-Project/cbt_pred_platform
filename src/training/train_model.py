"""
Model Training for CBT Prediction

Two-stage training pipeline:
    1. Initial Training: Train on external PROSPIE dataset
    2. Fine-tuning: Adapt to personal collected data

Usage:
    # Initial training on PROSPIE
    python -m src.training.train_model --mode train --data-dir data/prospie_ready
    
    # Fine-tuning on collected data
    python -m src.training.train_model --mode finetune --data-dir data/processed --base-model models/prospie
    
Features:
    - 84 training features (excludes user_id, timestamp metadata)
    - XGBoost regressor with GPU support (optional)
    - Feature scaling with StandardScaler
    - Cross-validation with stratified splits
    - Early stopping support
    - Model versioning and artifact management
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
    XGBOOST_VERSION = xgb.__version__
except ImportError:
    HAS_XGBOOST = False
    XGBOOST_VERSION = None
    logger.warning("XGBoost not installed. Install with: pip install xgboost")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.transformations import FeatureTransformer


class TrainingMode(Enum):
    """Training mode enumeration."""
    TRAIN = "train"           # Initial training from scratch
    FINETUNE = "finetune"     # Fine-tune existing model
    EVALUATE = "evaluate"     # Evaluate only


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Model hyperparameters
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.05
    min_child_weight: int = 3
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1      # L1 regularization
    reg_lambda: float = 1.0     # L2 regularization
    gamma: float = 0.1          # Min loss reduction for split
    
    # Training settings
    test_size: float = 0.2
    random_state: int = 42
    scale_features: bool = True
    n_cv_folds: int = 5
    early_stopping_rounds: int = 20
    
    # Fine-tuning specific
    finetune_learning_rate: float = 0.01  # Lower LR for fine-tuning
    finetune_n_estimators: int = 50       # Fewer rounds for fine-tuning
    freeze_n_trees: int = 0               # Trees to freeze (keep unchanged)
    
    # GPU support
    use_gpu: bool = False
    gpu_id: int = 0
    
    def to_xgb_params(self, mode: TrainingMode = TrainingMode.TRAIN) -> Dict[str, Any]:
        """Convert config to XGBoost parameters."""
        params = {
            "n_estimators": self.finetune_n_estimators if mode == TrainingMode.FINETUNE else self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.finetune_learning_rate if mode == TrainingMode.FINETUNE else self.learning_rate,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "gamma": self.gamma,
            "random_state": self.random_state,
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "verbosity": 0,
        }
        
        if self.use_gpu and HAS_XGBOOST:
            params["tree_method"] = "gpu_hist"
            params["gpu_id"] = self.gpu_id
        else:
            params["tree_method"] = "hist"
        
        return params


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    
    train_mae: float = 0.0
    train_rmse: float = 0.0
    train_r2: float = 0.0
    test_mae: float = 0.0
    test_rmse: float = 0.0
    test_r2: float = 0.0
    cv_mae_mean: float = 0.0
    cv_mae_std: float = 0.0
    cv_rmse_mean: float = 0.0
    cv_rmse_std: float = 0.0
    cv_r2_mean: float = 0.0
    cv_r2_std: float = 0.0
    n_train_samples: int = 0
    n_test_samples: int = 0
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "train": {
                "mae": self.train_mae,
                "rmse": self.train_rmse,
                "r2": self.train_r2,
                "n_samples": self.n_train_samples
            },
            "test": {
                "mae": self.test_mae,
                "rmse": self.test_rmse,
                "r2": self.test_r2,
                "n_samples": self.n_test_samples
            },
            "cv": {
                "mae_mean": self.cv_mae_mean,
                "mae_std": self.cv_mae_std,
                "rmse_mean": self.cv_rmse_mean,
                "rmse_std": self.cv_rmse_std,
                "r2_mean": self.cv_r2_mean,
                "r2_std": self.cv_r2_std
            },
            "feature_importance": self.feature_importance
        }
    
    def summary(self) -> str:
        """Return formatted summary string."""
        return (
            f"Train - MAE: {self.train_mae:.4f}°C, RMSE: {self.train_rmse:.4f}°C, R²: {self.train_r2:.4f}\n"
            f"Test  - MAE: {self.test_mae:.4f}°C, RMSE: {self.test_rmse:.4f}°C, R²: {self.test_r2:.4f}\n"
            f"CV    - MAE: {self.cv_mae_mean:.4f}±{self.cv_mae_std:.4f}°C, R²: {self.cv_r2_mean:.4f}±{self.cv_r2_std:.4f}"
        )


class ModelTrainer:
    """
    XGBoost model trainer for CBT prediction.
    
    Supports:
        - Initial training on PROSPIE dataset
        - Fine-tuning on personal data
        - Cross-validation
        - Early stopping
        - Model versioning
    
    Usage:
        # Training
        trainer = ModelTrainer()
        trainer.train(X, y)
        trainer.save("models/prospie")
        
        # Fine-tuning
        trainer = ModelTrainer.load("models/prospie")
        trainer.finetune(X_personal, y_personal)
        trainer.save("models/personal")
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize ModelTrainer.
        
        Args:
            config: Training configuration
        """
        if not HAS_XGBOOST:
            raise ImportError(
                "XGBoost is required for training. Install with: pip install xgboost"
            )
        
        self.config = config or TrainingConfig()
        self.model: Optional[xgb.XGBRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.metrics: Optional[TrainingMetrics] = None
        
        # Expected features from transformer
        self.expected_features = FeatureTransformer.get_training_features()
        
        logger.info(f"ModelTrainer initialized (XGBoost {XGBOOST_VERSION})")
    
    def _load_data(self, data_dir: Union[str, Path]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load features and labels from directory.
        
        Args:
            data_dir: Directory containing features.parquet and labels.parquet
            
        Returns:
            Tuple of (X, y)
        """
        data_dir = Path(data_dir)
        
        features_path = data_dir / "features.parquet"
        labels_path = data_dir / "labels.parquet"
        
        if not features_path.exists():
            raise FileNotFoundError(f"Features not found: {features_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels not found: {labels_path}")
        
        X = pd.read_parquet(features_path)
        y_df = pd.read_parquet(labels_path)
        
        # Extract target column
        target_cols = ["cbt_celsius", "cbt_temperature", "cbt"]
        y = None
        for col in target_cols:
            if col in y_df.columns:
                y = y_df[col]
                break
        
        if y is None:
            y = y_df.iloc[:, 0]
        
        logger.info(f"Loaded {len(X):,} samples from {data_dir}")
        
        return X, y
    
    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for training.
        
        Removes metadata columns and validates feature presence.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with training features only
        """
        metadata_cols = ["user_id", "timestamp"]
        feature_cols = [c for c in X.columns if c not in metadata_cols]
        
        # Check for missing features
        missing = [f for f in self.expected_features if f not in feature_cols]
        if missing:
            logger.warning(f"Missing {len(missing)} expected features: {missing[:5]}...")
        
        # Check for extra features
        extra = [f for f in feature_cols if f not in self.expected_features]
        if extra:
            logger.warning(f"Dropping {len(extra)} extra features: {extra[:5]}...")
            feature_cols = [c for c in feature_cols if c in self.expected_features]
        
        return X[feature_cols]
    
    def _handle_missing_values(
        self, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
        """
        Handle missing values using training set statistics.
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            Tuple of (X_train_filled, X_test_filled, fill_values)
        """
        fill_values = {}
        
        for col in X_train.columns:
            if X_train[col].isna().any() or X_test[col].isna().any():
                median_val = X_train[col].median()
                fill_values[col] = float(median_val)
                X_train[col] = X_train[col].fillna(median_val)
                X_test[col] = X_test[col].fillna(median_val)
        
        if fill_values:
            logger.info(f"Filled missing values in {len(fill_values)} columns")
        
        return X_train, X_test, fill_values
    
    def _compute_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray
    ) -> Tuple[float, float, float]:
        """Compute MAE, RMSE, and R² metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return mae, rmse, r2
    
    def _cross_validate(
        self,
        X: np.ndarray,
        y: pd.Series,
        params: Dict[str, Any]
    ) -> Dict[str, List[float]]:
        """
        Perform k-fold cross-validation.
        
        Args:
            X: Scaled feature array
            y: Target values
            params: XGBoost parameters
            
        Returns:
            Dictionary with fold scores
        """
        kf = KFold(
            n_splits=self.config.n_cv_folds,
            shuffle=True,
            random_state=self.config.random_state
        )
        
        scores = {"mae": [], "rmse": [], "r2": []}
        
        for fold, (train_idx, val_idx) in enumerate(
            tqdm(kf.split(X), total=self.config.n_cv_folds, desc="CV Folds", disable=not HAS_TQDM)
        ):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            
            fold_model = xgb.XGBRegressor(**params)
            fold_model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                verbose=False
            )
            
            y_pred = fold_model.predict(X_fold_val)
            mae, rmse, r2 = self._compute_metrics(y_fold_val, y_pred)
            
            scores["mae"].append(mae)
            scores["rmse"].append(rmse)
            scores["r2"].append(r2)
        
        return scores
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Extract feature importance from trained model."""
        if self.model is None:
            return {}
        
        importance = self.model.feature_importances_
        importance_dict = dict(zip(self.feature_names, importance))
        
        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def train(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        data_dir: Optional[Union[str, Path]] = None,
        run_cv: bool = True
    ) -> TrainingMetrics:
        """
        Train model from scratch on PROSPIE data.
        
        Args:
            X: Feature DataFrame (optional if data_dir provided)
            y: Target Series (optional if data_dir provided)
            data_dir: Directory with features.parquet and labels.parquet
            run_cv: Whether to run cross-validation
            
        Returns:
            TrainingMetrics with results
        """
        print("=" * 70)
        print("INITIAL MODEL TRAINING (PROSPIE Dataset)")
        print("=" * 70)
        print()
        
        # Load data if paths provided
        if X is None or y is None:
            if data_dir is None:
                data_dir = Path("data/prospie_ready")
            X, y = self._load_data(data_dir)
        
        # Prepare features
        logger.info("Preparing features...")
        X_prepared = self._prepare_features(X)
        self.feature_names = list(X_prepared.columns)
        
        print(f"Dataset: {len(y):,} samples × {len(self.feature_names)} features")
        print(f"Target range: {y.min():.2f}°C - {y.max():.2f}°C (mean: {y.mean():.2f}°C)")
        print()
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_prepared, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        
        print(f"Train set: {len(X_train):,} samples")
        print(f"Test set:  {len(X_test):,} samples")
        print()
        
        # Handle missing values
        X_train, X_test, fill_values = self._handle_missing_values(X_train.copy(), X_test.copy())
        self.metadata["fill_values"] = fill_values
        
        # Scale features
        if self.config.scale_features:
            logger.info("Scaling features...")
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values
        
        # Get XGBoost parameters
        params = self.config.to_xgb_params(TrainingMode.TRAIN)
        
        print("Model Configuration:")
        print(f"  n_estimators:    {params['n_estimators']}")
        print(f"  max_depth:       {params['max_depth']}")
        print(f"  learning_rate:   {params['learning_rate']}")
        print(f"  tree_method:     {params['tree_method']}")
        print()
        
        # Create and train model
        logger.info("Training XGBoost model...")
        print("Training model...")
        
        self.model = xgb.XGBRegressor(**params)
        
        eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]
        
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        print("✓ Training complete!")
        print()
        
        # Evaluate
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        train_mae, train_rmse, train_r2 = self._compute_metrics(y_train, y_train_pred)
        test_mae, test_rmse, test_r2 = self._compute_metrics(y_test, y_test_pred)
        
        # Initialize metrics
        self.metrics = TrainingMetrics(
            train_mae=train_mae,
            train_rmse=train_rmse,
            train_r2=train_r2,
            test_mae=test_mae,
            test_rmse=test_rmse,
            test_r2=test_r2,
            n_train_samples=len(y_train),
            n_test_samples=len(y_test)
        )
        
        print("Results:")
        print("┌" + "─" * 68 + "┐")
        print(f"│ {'Dataset':<10} {'MAE':>12} {'RMSE':>12} {'R²':>12} {'Samples':>15} │")
        print("├" + "─" * 68 + "┤")
        print(f"│ {'Train':<10} {train_mae:>10.4f}°C {train_rmse:>10.4f}°C {train_r2:>12.4f} {len(y_train):>15,} │")
        print(f"│ {'Test':<10} {test_mae:>10.4f}°C {test_rmse:>10.4f}°C {test_r2:>12.4f} {len(y_test):>15,} │")
        print("└" + "─" * 68 + "┘")
        print()
        
        # Cross-validation
        if run_cv:
            print(f"Running {self.config.n_cv_folds}-fold cross-validation...")
            cv_scores = self._cross_validate(X_train_scaled, y_train, params)
            
            self.metrics.cv_mae_mean = float(np.mean(cv_scores["mae"]))
            self.metrics.cv_mae_std = float(np.std(cv_scores["mae"]))
            self.metrics.cv_rmse_mean = float(np.mean(cv_scores["rmse"]))
            self.metrics.cv_rmse_std = float(np.std(cv_scores["rmse"]))
            self.metrics.cv_r2_mean = float(np.mean(cv_scores["r2"]))
            self.metrics.cv_r2_std = float(np.std(cv_scores["r2"]))
            
            print(f"  CV MAE:  {self.metrics.cv_mae_mean:.4f}°C (±{self.metrics.cv_mae_std:.4f})")
            print(f"  CV RMSE: {self.metrics.cv_rmse_mean:.4f}°C (±{self.metrics.cv_rmse_std:.4f})")
            print(f"  CV R²:   {self.metrics.cv_r2_mean:.4f} (±{self.metrics.cv_r2_std:.4f})")
            print()
        
        # Feature importance
        self.metrics.feature_importance = self._get_feature_importance()
        
        print("Top 10 Most Important Features:")
        for i, (name, imp) in enumerate(list(self.metrics.feature_importance.items())[:10]):
            bar = "█" * int(imp * 40)
            print(f"  {i+1:2}. {name:<35} {imp:.4f} {bar}")
        print()
        
        # Store metadata
        self.metadata.update({
            "mode": "train",
            "training_timestamp": datetime.now().isoformat(),
            "xgboost_version": XGBOOST_VERSION,
            "config": self.config.__dict__,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "metrics": self.metrics.to_dict(),
            "target_stats": {
                "mean": float(y.mean()),
                "std": float(y.std()),
                "min": float(y.min()),
                "max": float(y.max())
            }
        })
        
        print("=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        
        return self.metrics
    
    def finetune(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        data_dir: Optional[Union[str, Path]] = None,
        run_cv: bool = True
    ) -> TrainingMetrics:
        """
        Fine-tune existing model on personal data.
        
        Uses lower learning rate and fewer iterations to adapt
        the pre-trained model to personal data patterns.
        
        Args:
            X: Feature DataFrame (optional if data_dir provided)
            y: Target Series (optional if data_dir provided)
            data_dir: Directory with features.parquet and labels.parquet
            run_cv: Whether to run cross-validation
            
        Returns:
            TrainingMetrics with results
        """
        if self.model is None:
            raise ValueError(
                "No base model loaded. Call train() first or load a pre-trained model."
            )
        
        print("=" * 70)
        print("FINE-TUNING MODEL (Personal Data)")
        print("=" * 70)
        print()
        
        # Load data if paths provided
        if X is None or y is None:
            if data_dir is None:
                data_dir = Path("data/processed")
            X, y = self._load_data(data_dir)
        
        # Prepare features
        logger.info("Preparing features for fine-tuning...")
        X_prepared = self._prepare_features(X)
        
        # Verify feature alignment
        if list(X_prepared.columns) != self.feature_names:
            logger.warning("Feature order mismatch. Reordering columns...")
            X_prepared = X_prepared[self.feature_names]
        
        print(f"Fine-tuning dataset: {len(y):,} samples × {len(self.feature_names)} features")
        print(f"Target range: {y.min():.2f}°C - {y.max():.2f}°C (mean: {y.mean():.2f}°C)")
        print()
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_prepared, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        
        print(f"Train set: {len(X_train):,} samples")
        print(f"Test set:  {len(X_test):,} samples")
        print()
        
        # Handle missing values
        fill_values = self.metadata.get("fill_values", {})
        X_train = X_train.copy()
        X_test = X_test.copy()
        
        for col in X_train.columns:
            if X_train[col].isna().any() or X_test[col].isna().any():
                fill_val = fill_values.get(col, X_train[col].median())
                X_train[col] = X_train[col].fillna(fill_val)
                X_test[col] = X_test[col].fillna(fill_val)
        
        # Scale features using existing scaler
        if self.scaler is not None:
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values
        
        # Get fine-tuning parameters
        params = self.config.to_xgb_params(TrainingMode.FINETUNE)
        
        print("Fine-tuning Configuration:")
        print(f"  n_estimators:    {params['n_estimators']} (additional trees)")
        print(f"  learning_rate:   {params['learning_rate']} (reduced for fine-tuning)")
        print()
        
        # Store pre-fine-tune performance
        y_test_pred_before = self.model.predict(X_test_scaled)
        before_mae, before_rmse, before_r2 = self._compute_metrics(y_test, y_test_pred_before)
        
        print(f"Before fine-tuning - Test MAE: {before_mae:.4f}°C, R²: {before_r2:.4f}")
        print()
        
        # Fine-tune using xgb_model parameter for continuation
        logger.info("Fine-tuning model...")
        print("Fine-tuning model...")
        
        # Create new model that continues training from existing one
        finetuned_model = xgb.XGBRegressor(**params)
        
        eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]
        
        finetuned_model.fit(
            X_train_scaled, y_train,
            eval_set=eval_set,
            xgb_model=self.model.get_booster(),  # Continue from pre-trained model
            verbose=False
        )
        
        self.model = finetuned_model
        
        print("✓ Fine-tuning complete!")
        print()
        
        # Evaluate
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        train_mae, train_rmse, train_r2 = self._compute_metrics(y_train, y_train_pred)
        test_mae, test_rmse, test_r2 = self._compute_metrics(y_test, y_test_pred)
        
        # Calculate improvement
        mae_improvement = before_mae - test_mae
        r2_improvement = test_r2 - before_r2
        
        # Initialize metrics
        self.metrics = TrainingMetrics(
            train_mae=train_mae,
            train_rmse=train_rmse,
            train_r2=train_r2,
            test_mae=test_mae,
            test_rmse=test_rmse,
            test_r2=test_r2,
            n_train_samples=len(y_train),
            n_test_samples=len(y_test)
        )
        
        print("Results:")
        print("┌" + "─" * 68 + "┐")
        print(f"│ {'Dataset':<10} {'MAE':>12} {'RMSE':>12} {'R²':>12} {'Samples':>15} │")
        print("├" + "─" * 68 + "┤")
        print(f"│ {'Train':<10} {train_mae:>10.4f}°C {train_rmse:>10.4f}°C {train_r2:>12.4f} {len(y_train):>15,} │")
        print(f"│ {'Test':<10} {test_mae:>10.4f}°C {test_rmse:>10.4f}°C {test_r2:>12.4f} {len(y_test):>15,} │")
        print("└" + "─" * 68 + "┘")
        print()
        
        print("Improvement from fine-tuning:")
        print(f"  MAE:  {mae_improvement:+.4f}°C ({'better' if mae_improvement > 0 else 'worse'})")
        print(f"  R²:   {r2_improvement:+.4f} ({'better' if r2_improvement > 0 else 'worse'})")
        print()
        
        # Cross-validation on fine-tuned model
        if run_cv:
            print(f"Running {self.config.n_cv_folds}-fold cross-validation...")
            cv_scores = self._cross_validate(X_train_scaled, y_train, params)
            
            self.metrics.cv_mae_mean = float(np.mean(cv_scores["mae"]))
            self.metrics.cv_mae_std = float(np.std(cv_scores["mae"]))
            self.metrics.cv_rmse_mean = float(np.mean(cv_scores["rmse"]))
            self.metrics.cv_rmse_std = float(np.std(cv_scores["rmse"]))
            self.metrics.cv_r2_mean = float(np.mean(cv_scores["r2"]))
            self.metrics.cv_r2_std = float(np.std(cv_scores["r2"]))
            
            print(f"  CV MAE:  {self.metrics.cv_mae_mean:.4f}°C (±{self.metrics.cv_mae_std:.4f})")
            print(f"  CV RMSE: {self.metrics.cv_rmse_mean:.4f}°C (±{self.metrics.cv_rmse_std:.4f})")
            print(f"  CV R²:   {self.metrics.cv_r2_mean:.4f} (±{self.metrics.cv_r2_std:.4f})")
            print()
        
        # Feature importance
        self.metrics.feature_importance = self._get_feature_importance()
        
        # Update metadata
        self.metadata.update({
            "mode": "finetune",
            "finetune_timestamp": datetime.now().isoformat(),
            "finetune_metrics": self.metrics.to_dict(),
            "pre_finetune_test_mae": float(before_mae),
            "pre_finetune_test_r2": float(before_r2),
            "mae_improvement": float(mae_improvement),
            "r2_improvement": float(r2_improvement)
        })
        
        print("=" * 70)
        print("FINE-TUNING COMPLETE")
        print("=" * 70)
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predicted CBT values (Celsius)
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        # Prepare features
        X_prepared = self._prepare_features(X)
        
        # Ensure correct column order
        X_prepared = X_prepared[self.feature_names]
        
        # Handle missing values
        fill_values = self.metadata.get("fill_values", {})
        for col in X_prepared.columns:
            if X_prepared[col].isna().any():
                fill_val = fill_values.get(col, X_prepared[col].median())
                X_prepared[col] = X_prepared[col].fillna(fill_val)
        
        # Scale
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_prepared)
        else:
            X_scaled = X_prepared.values
        
        return self.model.predict(X_scaled)
    
    def save(self, output_dir: Union[str, Path]) -> Dict[str, str]:
        """
        Save model and all artifacts.
        
        Saves:
            - model.json: XGBoost model (JSON format for portability)
            - model.joblib: XGBoost model (pickle for fast loading)
            - scaler.joblib: Feature scaler
            - metadata.json: Training configuration and metrics
            - feature_names.json: Feature list
            
        Args:
            output_dir: Directory to save artifacts
            
        Returns:
            Dictionary with saved file paths
        """
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        print(f"\nSaving model to: {output_dir}")
        
        # Save XGBoost model in JSON format (portable)
        model_json_path = output_dir / "model.json"
        self.model.save_model(model_json_path)
        saved_files["model_json"] = str(model_json_path)
        
        # Save XGBoost model with joblib (faster loading)
        model_joblib_path = output_dir / "model.joblib"
        joblib.dump(self.model, model_joblib_path)
        saved_files["model_joblib"] = str(model_joblib_path)
        
        # Save scaler
        if self.scaler is not None:
            scaler_path = output_dir / "scaler.joblib"
            joblib.dump(self.scaler, scaler_path)
            saved_files["scaler"] = str(scaler_path)
        
        # Save metadata
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)
        saved_files["metadata"] = str(metadata_path)
        
        # Save feature names
        features_path = output_dir / "feature_names.json"
        with open(features_path, "w") as f:
            json.dump({
                "feature_names": self.feature_names,
                "n_features": len(self.feature_names)
            }, f, indent=2)
        saved_files["feature_names"] = str(features_path)
        
        print("Saved artifacts:")
        for name, path in saved_files.items():
            print(f"  ✓ {name}: {Path(path).name}")
        
        return saved_files
    
    @classmethod
    def load(cls, model_dir: Union[str, Path]) -> "ModelTrainer":
        """
        Load a trained model from disk.
        
        Args:
            model_dir: Directory containing saved model
            
        Returns:
            ModelTrainer instance with loaded model
        """
        model_dir = Path(model_dir)
        
        print(f"Loading model from: {model_dir}")
        
        trainer = cls()
        
        # Load model (prefer joblib for speed)
        model_joblib_path = model_dir / "model.joblib"
        model_json_path = model_dir / "model.json"
        
        if model_joblib_path.exists():
            trainer.model = joblib.load(model_joblib_path)
        elif model_json_path.exists():
            trainer.model = xgb.XGBRegressor()
            trainer.model.load_model(model_json_path)
        else:
            raise FileNotFoundError(f"No model file found in {model_dir}")
        
        # Load scaler
        scaler_path = model_dir / "scaler.joblib"
        if scaler_path.exists():
            trainer.scaler = joblib.load(scaler_path)
        
        # Load metadata
        metadata_path = model_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                trainer.metadata = json.load(f)
        
        # Load feature names
        features_path = model_dir / "feature_names.json"
        if features_path.exists():
            with open(features_path, "r") as f:
                data = json.load(f)
                trainer.feature_names = data.get("feature_names", [])
        
        print(f"  ✓ Model loaded: {trainer.model.__class__.__name__}")
        print(f"  ✓ Features: {len(trainer.feature_names)}")
        
        if trainer.metadata.get("metrics"):
            test_mae = trainer.metadata["metrics"].get("test", {}).get("mae", "N/A")
            print(f"  ✓ Test MAE: {test_mae}")
        
        return trainer


def main():
    """Command-line interface for training."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train or fine-tune CBT prediction model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on PROSPIE data
  python -m src.training.train_model --mode train --data-dir data/prospie_ready --output models/prospie

  # Fine-tune on personal data
  python -m src.training.train_model --mode finetune --data-dir data/processed --base-model models/prospie --output models/personal

  # Train with custom hyperparameters
  python -m src.training.train_model --mode train --n-estimators 300 --max-depth 8 --learning-rate 0.03
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "finetune"],
        default="train",
        help="Training mode: 'train' for initial training, 'finetune' for adaptation"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory with features.parquet and labels.parquet"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Path to base model for fine-tuning"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models",
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of boosting rounds"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum tree depth"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate"
    )
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Skip cross-validation"
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for training (requires CUDA)"
    )
    
    args = parser.parse_args()
    
    # Set default data directories based on mode
    if args.data_dir is None:
        if args.mode == "train":
            args.data_dir = "data/prospie_ready"
        else:
            args.data_dir = "data/processed"
    
    # Create config
    config = TrainingConfig(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        use_gpu=args.use_gpu
    )
    
    # Execute based on mode
    if args.mode == "train":
        trainer = ModelTrainer(config)
        trainer.train(data_dir=args.data_dir, run_cv=not args.no_cv)
        trainer.save(args.output)
        
    elif args.mode == "finetune":
        if args.base_model is None:
            args.base_model = "models/prospie"
        
        trainer = ModelTrainer.load(args.base_model)
        trainer.config = config
        trainer.finetune(data_dir=args.data_dir, run_cv=not args.no_cv)
        trainer.save(args.output)


if __name__ == "__main__":
    main()