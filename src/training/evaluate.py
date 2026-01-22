"""
Model Evaluation for CBT Prediction

Comprehensive evaluation framework for both initial training and fine-tuned models.
Includes:
    - Standard ML metrics (MAE, RMSE, R², MAPE)
    - Clinical accuracy thresholds
    - Error analysis by feature ranges
    - Cross-domain evaluation (PROSPIE vs Personal)
    - Visualization suite
    - Model comparison reports

Usage:
    # Evaluate single model
    python -m src.training.evaluate --model-dir models/prospie --data-dir data/prospie_ready

    # Compare base vs fine-tuned
    python -m src.training.evaluate --compare --base-model models/prospie --finetuned-model models/personal

Author: CBT Prediction Platform
Date: January 2026
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    
    # Core regression metrics
    mae: float = 0.0
    rmse: float = 0.0
    r2: float = 0.0
    mape: float = 0.0
    
    # Error statistics
    mean_error: float = 0.0
    std_error: float = 0.0
    max_error: float = 0.0
    min_error: float = 0.0
    
    # Percentiles
    error_p10: float = 0.0
    error_p25: float = 0.0
    error_p50: float = 0.0
    error_p75: float = 0.0
    error_p90: float = 0.0
    error_p95: float = 0.0
    error_p99: float = 0.0
    
    # Clinical accuracy
    within_0_1c: float = 0.0  # Percentage within +/-0.1C
    within_0_2c: float = 0.0  # Percentage within +/-0.2C
    within_0_3c: float = 0.0  # Percentage within +/-0.3C
    within_0_5c: float = 0.0  # Percentage within +/-0.5C
    
    # Sample info
    n_samples: int = 0
    dataset_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "regression_metrics": {
                "mae": self.mae,
                "rmse": self.rmse,
                "r2": self.r2,
                "mape": self.mape
            },
            "error_statistics": {
                "mean": self.mean_error,
                "std": self.std_error,
                "max": self.max_error,
                "min": self.min_error
            },
            "error_percentiles": {
                "p10": self.error_p10,
                "p25": self.error_p25,
                "p50": self.error_p50,
                "p75": self.error_p75,
                "p90": self.error_p90,
                "p95": self.error_p95,
                "p99": self.error_p99
            },
            "clinical_accuracy": {
                "within_0.1C": self.within_0_1c,
                "within_0.2C": self.within_0_2c,
                "within_0.3C": self.within_0_3c,
                "within_0.5C": self.within_0_5c
            },
            "n_samples": self.n_samples,
            "dataset": self.dataset_name
        }
    
    def summary(self) -> str:
        """Return formatted summary string (ASCII-safe for Windows)."""
        return (
            f"Dataset: {self.dataset_name} ({self.n_samples:,} samples)\n"
            f"-" * 50 + "\n"
            f"Regression Metrics:\n"
            f"  MAE:  {self.mae:.4f}C\n"
            f"  RMSE: {self.rmse:.4f}C\n"
            f"  R2:   {self.r2:.4f}\n"
            f"  MAPE: {self.mape:.2f}%\n"
            f"\n"
            f"Clinical Accuracy:\n"
            f"  Within +/-0.1C: {self.within_0_1c:.1f}%\n"
            f"  Within +/-0.2C: {self.within_0_2c:.1f}%\n"
            f"  Within +/-0.3C: {self.within_0_3c:.1f}%\n"
            f"  Within +/-0.5C: {self.within_0_5c:.1f}%\n"
            f"\n"
            f"Error Percentiles:\n"
            f"  50th (median): {self.error_p50:.4f}C\n"
            f"  90th:          {self.error_p90:.4f}C\n"
            f"  95th:          {self.error_p95:.4f}C"
        )


@dataclass
class ComparisonReport:
    """Comparison between base and fine-tuned models."""
    
    base_metrics: EvaluationMetrics = None
    finetuned_metrics: EvaluationMetrics = None
    improvement: Dict[str, float] = field(default_factory=dict)
    
    def compute_improvement(self) -> None:
        """Compute improvement percentages."""
        if self.base_metrics is None or self.finetuned_metrics is None:
            return
        
        # Lower is better for MAE, RMSE
        if self.base_metrics.mae != 0:
            self.improvement["mae"] = (
                (self.base_metrics.mae - self.finetuned_metrics.mae) / self.base_metrics.mae * 100
            )
        else:
            self.improvement["mae"] = 0.0
            
        if self.base_metrics.rmse != 0:
            self.improvement["rmse"] = (
                (self.base_metrics.rmse - self.finetuned_metrics.rmse) / self.base_metrics.rmse * 100
            )
        else:
            self.improvement["rmse"] = 0.0
        
        # Higher is better for R2
        if self.base_metrics.r2 != 0:
            self.improvement["r2"] = (
                (self.finetuned_metrics.r2 - self.base_metrics.r2) / abs(self.base_metrics.r2) * 100
            )
        else:
            self.improvement["r2"] = 0.0
        
        # Clinical accuracy improvements
        self.improvement["within_0.3C"] = (
            self.finetuned_metrics.within_0_3c - self.base_metrics.within_0_3c
        )


class ModelEvaluator:
    """
    Comprehensive model evaluation for CBT prediction.
    
    Supports:
        - Single model evaluation
        - Base vs fine-tuned comparison
        - Cross-domain evaluation
        - Error analysis
        - Visualization
    
    Usage:
        evaluator = ModelEvaluator()
        evaluator.load_model("models/personal")
        metrics = evaluator.evaluate(X, y)
        evaluator.save_report("evaluation_results/")
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.metadata: Dict[str, Any] = {}
        
        self.predictions: Optional[np.ndarray] = None
        self.y_true: Optional[np.ndarray] = None
        self.errors: Optional[np.ndarray] = None
        
        self.metrics: Optional[EvaluationMetrics] = None
        self.comparison: Optional[ComparisonReport] = None
        
        # Set style
        if HAS_SEABORN:
            sns.set_style("whitegrid")
            sns.set_palette("husl")
    
    def load_model(self, model_dir: Union[str, Path]) -> None:
        """Load model from directory."""
        model_dir = Path(model_dir)
        
        # Load model
        model_path = model_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = joblib.load(model_path)
        
        # Load scaler
        scaler_path = model_dir / "scaler.joblib"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        
        # Load feature names
        features_path = model_dir / "feature_names.json"
        if features_path.exists():
            with open(features_path, encoding="utf-8") as f:
                self.feature_names = json.load(f).get("feature_names", [])
        
        # Load metadata
        metadata_path = model_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                self.metadata = json.load(f)
        
        logger.info(f"Loaded model from {model_dir}")
    
    def load_data(
        self, 
        data_dir: Union[str, Path]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Load evaluation data."""
        data_dir = Path(data_dir)
        
        X = pd.read_parquet(data_dir / "features.parquet")
        y_df = pd.read_parquet(data_dir / "labels.parquet")
        
        # Extract target
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
    
    def _prepare_features(self, X: pd.DataFrame) -> np.ndarray:
        """Prepare features for prediction."""
        # Remove metadata columns
        metadata_cols = ["user_id", "timestamp"]
        feature_cols = [c for c in X.columns if c not in metadata_cols]
        X_features = X[feature_cols].copy()
        
        # Align to expected features
        if self.feature_names:
            for col in self.feature_names:
                if col not in X_features.columns:
                    X_features[col] = 0
            X_features = X_features[self.feature_names]
        
        # Fill missing
        X_features = X_features.fillna(0)
        
        # Scale
        if self.scaler is not None:
            return self.scaler.transform(X_features)
        return X_features.values
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str = "Test"
    ) -> EvaluationMetrics:
        """
        Evaluate model on dataset.
        
        Args:
            X: Feature DataFrame
            y: True labels
            dataset_name: Name for reporting
            
        Returns:
            EvaluationMetrics object
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        print("=" * 60)
        print(f"MODEL EVALUATION: {dataset_name}")
        print("=" * 60)
        print()
        
        # Prepare and predict
        X_prepared = self._prepare_features(X)
        self.predictions = self.model.predict(X_prepared)
        self.y_true = y.values
        self.errors = self.predictions - self.y_true
        
        # Compute metrics
        abs_errors = np.abs(self.errors)
        
        self.metrics = EvaluationMetrics(
            # Regression metrics
            mae=float(mean_absolute_error(y, self.predictions)),
            rmse=float(np.sqrt(mean_squared_error(y, self.predictions))),
            r2=float(r2_score(y, self.predictions)),
            mape=float(mean_absolute_percentage_error(y, self.predictions) * 100),
            
            # Error statistics
            mean_error=float(np.mean(self.errors)),
            std_error=float(np.std(self.errors)),
            max_error=float(np.max(abs_errors)),
            min_error=float(np.min(abs_errors)),
            
            # Percentiles
            error_p10=float(np.percentile(abs_errors, 10)),
            error_p25=float(np.percentile(abs_errors, 25)),
            error_p50=float(np.percentile(abs_errors, 50)),
            error_p75=float(np.percentile(abs_errors, 75)),
            error_p90=float(np.percentile(abs_errors, 90)),
            error_p95=float(np.percentile(abs_errors, 95)),
            error_p99=float(np.percentile(abs_errors, 99)),
            
            # Clinical accuracy
            within_0_1c=float(np.mean(abs_errors <= 0.1) * 100),
            within_0_2c=float(np.mean(abs_errors <= 0.2) * 100),
            within_0_3c=float(np.mean(abs_errors <= 0.3) * 100),
            within_0_5c=float(np.mean(abs_errors <= 0.5) * 100),
            
            # Info
            n_samples=len(y),
            dataset_name=dataset_name
        )
        
        print(self.metrics.summary())
        print()
        
        return self.metrics
    
    def compare_models(
        self,
        base_model_dir: Union[str, Path],
        finetuned_model_dir: Union[str, Path],
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str = "Personal Data"
    ) -> ComparisonReport:
        """
        Compare base model vs fine-tuned model.
        
        Args:
            base_model_dir: Path to base (PROSPIE) model
            finetuned_model_dir: Path to fine-tuned model
            X: Evaluation features
            y: True labels
            dataset_name: Dataset description
            
        Returns:
            ComparisonReport with both metrics and improvement
        """
        print("=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        print()
        
        # Evaluate base model
        print("Evaluating BASE model (PROSPIE-trained)...")
        print("-" * 40)
        self.load_model(base_model_dir)
        base_metrics = self.evaluate(X, y, f"{dataset_name} - Base Model")
        
        print()
        
        # Evaluate fine-tuned model
        print("Evaluating FINE-TUNED model...")
        print("-" * 40)
        self.load_model(finetuned_model_dir)
        ft_metrics = self.evaluate(X, y, f"{dataset_name} - Fine-tuned")
        
        # Build comparison
        self.comparison = ComparisonReport(
            base_metrics=base_metrics,
            finetuned_metrics=ft_metrics
        )
        self.comparison.compute_improvement()
        
        print()
        print("=" * 60)
        print("IMPROVEMENT SUMMARY")
        print("=" * 60)
        print()
        
        imp = self.comparison.improvement
        print(f"MAE Improvement:  {imp['mae']:+.2f}% ({'BETTER' if imp['mae'] > 0 else 'WORSE'})")
        print(f"RMSE Improvement: {imp['rmse']:+.2f}% ({'BETTER' if imp['rmse'] > 0 else 'WORSE'})")
        print(f"R2 Improvement:   {imp['r2']:+.2f}% ({'BETTER' if imp['r2'] > 0 else 'WORSE'})")
        print(f"Clinical (+/-0.3C): {imp['within_0.3C']:+.1f} percentage points")
        print()
        
        return self.comparison
    
    def analyze_errors_by_range(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> pd.DataFrame:
        """Analyze errors by CBT range."""
        if self.predictions is None:
            self.evaluate(X, y)
        
        # Create bins
        bins = [35, 36, 36.5, 37, 37.5, 38, 42]
        labels = ["<36", "36-36.5", "36.5-37", "37-37.5", "37.5-38", ">38"]
        
        df = pd.DataFrame({
            "y_true": self.y_true,
            "y_pred": self.predictions,
            "abs_error": np.abs(self.errors)
        })
        df["range"] = pd.cut(df["y_true"], bins=bins, labels=labels)
        
        analysis = df.groupby("range").agg({
            "abs_error": ["count", "mean", "std", "max"],
            "y_true": "mean"
        }).round(4)
        
        print("\nError Analysis by CBT Range:")
        print(analysis.to_string())
        
        return analysis
    
    def plot_evaluation_suite(
        self,
        output_dir: Union[str, Path],
        show: bool = False
    ) -> List[str]:
        """
        Generate comprehensive evaluation plots.
        
        Creates:
            - Predicted vs Actual scatter plot
            - Error distribution histogram
            - Residuals plot
            - Clinical accuracy bar chart
            - Feature importance (if available)
        
        Returns:
            List of saved file paths
        """
        if not HAS_MATPLOTLIB:
            logger.warning("Matplotlib not installed. Skipping plots.")
            return []
        
        if self.predictions is None:
            raise ValueError("No predictions. Call evaluate() first.")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        # Set up style
        plt.rcParams.update({
            "figure.figsize": (10, 6),
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12
        })
        
        # 1. Predicted vs Actual
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.scatter(self.y_true, self.predictions, alpha=0.5, s=20, edgecolors='none')
        
        # Perfect prediction line
        min_val = min(self.y_true.min(), self.predictions.min()) - 0.2
        max_val = max(self.y_true.max(), self.predictions.max()) + 0.2
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # +/-0.3C confidence band
        ax.fill_between(
            [min_val, max_val],
            [min_val - 0.3, max_val - 0.3],
            [min_val + 0.3, max_val + 0.3],
            alpha=0.15, color='green', label='+/-0.3C'
        )
        
        ax.set_xlabel('Actual CBT (C)')
        ax.set_ylabel('Predicted CBT (C)')
        ax.set_title(f'Predicted vs Actual CBT\nMAE: {self.metrics.mae:.3f}C, R2: {self.metrics.r2:.3f}')
        ax.legend(loc='upper left')
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        path = output_dir / "01_predicted_vs_actual.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        saved_files.append(str(path))
        if show:
            plt.show()
        plt.close()
        
        # 2. Error Distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(self.errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].axvline(0, color='r', linestyle='--', lw=2, label='Zero')
        axes[0].axvline(self.errors.mean(), color='orange', lw=2, 
                       label=f'Mean: {self.errors.mean():.3f}C')
        axes[0].axvline(-0.3, color='green', linestyle=':', lw=1.5, alpha=0.7)
        axes[0].axvline(0.3, color='green', linestyle=':', lw=1.5, alpha=0.7, label='+/-0.3C')
        axes[0].set_xlabel('Prediction Error (C)')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Error Distribution')
        axes[0].legend()
        
        # Box plot of absolute errors
        abs_errors = np.abs(self.errors)
        axes[1].boxplot(abs_errors, vert=True, widths=0.5)
        axes[1].axhline(0.3, color='green', linestyle='--', lw=2, label='0.3C threshold')
        axes[1].set_ylabel('Absolute Error (C)')
        axes[1].set_title(f'Absolute Error Distribution\nMedian: {np.median(abs_errors):.3f}C')
        axes[1].set_xticklabels(['All Samples'])
        axes[1].legend()
        
        plt.tight_layout()
        path = output_dir / "02_error_distribution.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        saved_files.append(str(path))
        if show:
            plt.show()
        plt.close()
        
        # 3. Residuals Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(self.predictions, self.errors, alpha=0.5, s=20, edgecolors='none')
        ax.axhline(0, color='r', linestyle='--', lw=2)
        ax.axhline(0.3, color='green', linestyle=':', lw=1.5, alpha=0.7)
        ax.axhline(-0.3, color='green', linestyle=':', lw=1.5, alpha=0.7)
        
        ax.set_xlabel('Predicted CBT (C)')
        ax.set_ylabel('Residual (Predicted - Actual) (C)')
        ax.set_title('Residuals vs Predicted Values')
        ax.grid(True, alpha=0.3)
        
        path = output_dir / "03_residuals.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        saved_files.append(str(path))
        if show:
            plt.show()
        plt.close()
        
        # 4. Clinical Accuracy Bar Chart
        fig, ax = plt.subplots(figsize=(8, 5))
        
        thresholds = ['+/-0.1C', '+/-0.2C', '+/-0.3C', '+/-0.5C']
        accuracies = [
            self.metrics.within_0_1c,
            self.metrics.within_0_2c,
            self.metrics.within_0_3c,
            self.metrics.within_0_5c
        ]
        
        colors = ['#e74c3c', '#f39c12', '#27ae60', '#2ecc71']
        bars = ax.bar(thresholds, accuracies, color=colors, edgecolor='black')
        
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Percentage of Predictions')
        ax.set_title('Clinical Accuracy Thresholds')
        ax.set_ylim(0, 105)
        ax.axhline(80, color='gray', linestyle='--', alpha=0.5, label='80% target')
        ax.legend()
        
        path = output_dir / "04_clinical_accuracy.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        saved_files.append(str(path))
        if show:
            plt.show()
        plt.close()
        
        # 5. Feature Importance (if available)
        if hasattr(self.model, 'feature_importances_') and self.feature_names:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            importance = self.model.feature_importances_
            n_features = min(20, len(importance))
            indices = np.argsort(importance)[-n_features:]
            
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(indices)))
            
            ax.barh(
                range(len(indices)),
                importance[indices],
                color=colors,
                edgecolor='black'
            )
            ax.set_yticks(range(len(indices)))
            
            # Safe indexing for feature names
            feature_labels = []
            for i in indices:
                if i < len(self.feature_names):
                    feature_labels.append(self.feature_names[i])
                else:
                    feature_labels.append(f"feature_{i}")
            
            ax.set_yticklabels(feature_labels)
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'Top {n_features} Most Important Features')
            
            plt.tight_layout()
            path = output_dir / "05_feature_importance.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            saved_files.append(str(path))
            if show:
                plt.show()
            plt.close()
        
        print(f"\nSaved {len(saved_files)} plots to {output_dir}")
        return saved_files
    
    def save_report(
        self,
        output_dir: Union[str, Path],
        include_plots: bool = True
    ) -> Dict[str, str]:
        """
        Save complete evaluation report.
        
        Saves:
            - evaluation_metrics.json
            - evaluation_summary.txt
            - All plots (if include_plots=True)
        
        Returns:
            Dictionary with saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save metrics JSON
        metrics_path = output_dir / "evaluation_metrics.json"
        report = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "model_metadata": self.metadata,
            "metrics": self.metrics.to_dict() if self.metrics else {}
        }
        
        if self.comparison:
            report["comparison"] = {
                "base_model": self.comparison.base_metrics.to_dict(),
                "finetuned_model": self.comparison.finetuned_metrics.to_dict(),
                "improvement": self.comparison.improvement
            }
        
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        saved_files["metrics_json"] = str(metrics_path)
        
        # Save text summary (use UTF-8 encoding)
        summary_path = output_dir / "evaluation_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("CBT PREDICTION MODEL EVALUATION REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if self.metrics:
                f.write(self.metrics.summary())
                f.write("\n\n")
            
            if self.comparison:
                f.write("MODEL COMPARISON\n")
                f.write("-" * 40 + "\n")
                f.write(f"MAE Improvement:  {self.comparison.improvement.get('mae', 0):+.2f}%\n")
                f.write(f"RMSE Improvement: {self.comparison.improvement.get('rmse', 0):+.2f}%\n")
                f.write(f"R2 Improvement:   {self.comparison.improvement.get('r2', 0):+.2f}%\n")
        
        saved_files["summary_txt"] = str(summary_path)
        
        # Save plots
        if include_plots and self.predictions is not None:
            plot_files = self.plot_evaluation_suite(output_dir)
            saved_files["plots"] = plot_files
        
        print(f"\nEvaluation report saved to: {output_dir}")
        return saved_files


def main():
    """Command-line interface for evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate CBT prediction model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate single model
  python -m src.training.evaluate --model-dir models/personal --data-dir data/processed

  # Compare base vs fine-tuned
  python -m src.training.evaluate --compare --base-model models/prospie --finetuned-model models/personal --data-dir data/processed

  # Evaluate on PROSPIE data
  python -m src.training.evaluate --model-dir models/prospie --data-dir data/prospie_ready
        """
    )
    
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/personal",
        help="Directory containing trained model"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing evaluation data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for evaluation results (default: model-dir/evaluation)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare base model vs fine-tuned model"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="models/prospie",
        help="Base model directory (for comparison)"
    )
    parser.add_argument(
        "--finetuned-model",
        type=str,
        default="models/personal",
        help="Fine-tuned model directory (for comparison)"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots"
    )
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        if args.compare:
            args.output_dir = Path(args.finetuned_model) / "evaluation"
        else:
            args.output_dir = Path(args.model_dir) / "evaluation"
    
    evaluator = ModelEvaluator()
    
    # Load data
    X, y = evaluator.load_data(args.data_dir)
    
    if args.compare:
        # Compare models
        evaluator.compare_models(
            args.base_model,
            args.finetuned_model,
            X, y
        )
    else:
        # Single model evaluation
        evaluator.load_model(args.model_dir)
        evaluator.evaluate(X, y)
        evaluator.analyze_errors_by_range(X, y)
    
    # Save report
    evaluator.save_report(args.output_dir, include_plots=not args.no_plots)


if __name__ == "__main__":
    main()