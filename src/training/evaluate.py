"""
Model Evaluation

Detailed evaluation and error analysis of the trained model.

Usage:
    python -m src.training.evaluate --model-dir models/ --data-dir data/processed/
"""

import json
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .train_model import ModelTrainer


class ModelEvaluator:
    """Evaluate trained CBT model."""
    
    def __init__(self, trainer: ModelTrainer):
        self.trainer = trainer
        self.results: Dict = {}
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        save_plots: bool = True,
        output_dir: Union[str, Path] = "models/"
    ) -> Dict:
        """
        Run full evaluation.
        
        Args:
            X: Features
            y: True CBT values
            save_plots: Whether to save plots
            output_dir: Where to save outputs
        
        Returns:
            Dict with evaluation results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        
        # Get predictions
        predictions = self.trainer.predict(X)
        
        # Basic metrics
        metrics = {
            "mae": mean_absolute_error(y, predictions),
            "rmse": np.sqrt(mean_squared_error(y, predictions)),
            "r2": r2_score(y, predictions),
            "max_error": np.max(np.abs(y - predictions)),
            "mean_prediction": np.mean(predictions),
            "std_prediction": np.std(predictions)
        }
        
        print(f"\nOverall Metrics:")
        print(f"  MAE:  {metrics['mae']:.4f}°C")
        print(f"  RMSE: {metrics['rmse']:.4f}°C")
        print(f"  R²:   {metrics['r2']:.4f}")
        print(f"  Max Error: {metrics['max_error']:.4f}°C")
        
        # Error distribution
        errors = predictions - y.values
        metrics["error_percentiles"] = {
            "p10": float(np.percentile(np.abs(errors), 10)),
            "p50": float(np.percentile(np.abs(errors), 50)),
            "p90": float(np.percentile(np.abs(errors), 90)),
            "p95": float(np.percentile(np.abs(errors), 95))
        }
        
        print(f"\nError Percentiles (absolute):")
        print(f"  10th: {metrics['error_percentiles']['p10']:.4f}°C")
        print(f"  50th: {metrics['error_percentiles']['p50']:.4f}°C")
        print(f"  90th: {metrics['error_percentiles']['p90']:.4f}°C")
        print(f"  95th: {metrics['error_percentiles']['p95']:.4f}°C")
        
        # Clinically relevant accuracy (within 0.3°C)
        within_03 = np.mean(np.abs(errors) <= 0.3) * 100
        within_05 = np.mean(np.abs(errors) <= 0.5) * 100
        
        metrics["clinical_accuracy"] = {
            "within_0.3C": float(within_03),
            "within_0.5C": float(within_05)
        }
        
        print(f"\nClinical Accuracy:")
        print(f"  Within ±0.3°C: {within_03:.1f}%")
        print(f"  Within ±0.5°C: {within_05:.1f}%")
        
        self.results = metrics
        
        # Save plots
        if save_plots:
            self._plot_predictions(y, predictions, output_dir)
            self._plot_error_distribution(errors, output_dir)
            self._plot_residuals(y, predictions, output_dir)
        
        # Save metrics
        with open(output_dir / "evaluation_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=float)
        print(f"\nSaved metrics to {output_dir / 'evaluation_metrics.json'}")
        
        return metrics
    
    def _plot_predictions(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        output_dir: Path
    ) -> None:
        """Plot predicted vs actual values."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.scatter(y_true, y_pred, alpha=0.5, edgecolors='none')
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
        
        # ±0.3°C bands
        ax.fill_between(
            [min_val, max_val],
            [min_val - 0.3, max_val - 0.3],
            [min_val + 0.3, max_val + 0.3],
            alpha=0.2, color='green', label='±0.3°C'
        )
        
        ax.set_xlabel('Actual CBT (°C)', fontsize=12)
        ax.set_ylabel('Predicted CBT (°C)', fontsize=12)
        ax.set_title('Predicted vs Actual CBT', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "predicted_vs_actual.png", dpi=150)
        plt.close()
        print(f"Saved plot: {output_dir / 'predicted_vs_actual.png'}")
    
    def _plot_error_distribution(
        self,
        errors: np.ndarray,
        output_dir: Path
    ) -> None:
        """Plot error distribution histogram."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(errors, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='r', linestyle='--', lw=2, label='Zero error')
        ax.axvline(errors.mean(), color='g', linestyle='-', lw=2, 
                   label=f'Mean: {errors.mean():.3f}°C')
        
        ax.set_xlabel('Prediction Error (°C)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Error Distribution', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "error_distribution.png", dpi=150)
        plt.close()
        print(f"Saved plot: {output_dir / 'error_distribution.png'}")
    
    def _plot_residuals(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        output_dir: Path
    ) -> None:
        """Plot residuals vs predicted values."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        residuals = y_true.values - y_pred
        
        ax.scatter(y_pred, residuals, alpha=0.5, edgecolors='none')
        ax.axhline(0, color='r', linestyle='--', lw=2)
        ax.axhline(0.3, color='g', linestyle=':', lw=1, alpha=0.7)
        ax.axhline(-0.3, color='g', linestyle=':', lw=1, alpha=0.7)
        
        ax.set_xlabel('Predicted CBT (°C)', fontsize=12)
        ax.set_ylabel('Residual (°C)', fontsize=12)
        ax.set_title('Residuals vs Predicted Values', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "residuals.png", dpi=150)
        plt.close()
        print(f"Saved plot: {output_dir / 'residuals.png'}")


def main():
    """Command-line interface for evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Directory containing trained model"
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Directory containing evaluation data"
    )
    
    args = parser.parse_args()
    
    # Load model
    trainer = ModelTrainer.load(args.model_dir)
    
    # Load data
    data_dir = Path(args.data_dir)
    X = pd.read_parquet(data_dir / "features.parquet")
    y = pd.read_parquet(data_dir / "labels.parquet")["cbt_temperature"]
    
    # Evaluate
    evaluator = ModelEvaluator(trainer)
    evaluator.evaluate(X, y, output_dir=args.model_dir)


if __name__ == "__main__":
    main()