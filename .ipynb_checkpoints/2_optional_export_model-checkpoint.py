#!/usr/bin/env python3
"""
Export Model from MLflow or Domino Model Registry

Export a trained model for deployment as a model API endpoint.

Usage:
    python export_model.py --metric accuracy
    python export_model.py --run-id abc123def456
    python export_model.py --output /mnt/artifacts/model.pkl
"""

import argparse
import sys
from pathlib import Path
import joblib
import mlflow


def find_best_run(metric="accuracy", experiment_name_pattern=None):
    """
    Find the best model run based on a metric.

    Args:
        metric: Metric to maximize
        experiment_name_pattern: Pattern to match experiment names (optional)

    Returns:
        run_id of the best model
    """
    experiments = mlflow.search_experiments()

    if experiment_name_pattern:
        experiments = [exp for exp in experiments if experiment_name_pattern in exp.name]

    if not experiments:
        raise ValueError("No experiments found")

    print(f"{'='*60}")
    print(f"SEARCHING FOR BEST MODEL")
    print(f"{'='*60}")
    print(f"Metric: {metric}")
    print(f"Experiments: {len(experiments)}")

    all_runs = []
    for exp in experiments:
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=10
        )
        if len(runs) > 0:
            all_runs.append(runs)

    if not all_runs:
        raise ValueError("No runs found")

    import pandas as pd
    combined_runs = pd.concat(all_runs, ignore_index=True)
    combined_runs = combined_runs.sort_values(f"metrics.{metric}", ascending=False)

    best_run = combined_runs.iloc[0]
    run_id = best_run['run_id']

    print(f"✅ Best model found:")
    print(f"   Run ID: {run_id}")
    print(f"   {metric}: {best_run[f'metrics.{metric}']:.4f}")

    return run_id


def export_model(run_id, output_path="/mnt/artifacts/model.pkl"):
    """
    Export a model from MLflow.

    Args:
        run_id: MLflow run ID
        output_path: Path to save the model

    Returns:
        Path to exported model
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"EXPORTING MODEL")
    print(f"{'='*60}")

    model_uri = f"runs:/{run_id}/model"
    print(f"Loading from: {model_uri}")

    try:
        model = mlflow.sklearn.load_model(model_uri)
        print(f"✅ Model loaded")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    print(f"Saving to: {output_path}")
    joblib.dump(model, output_path)

    if output_path.exists():
        size_kb = output_path.stat().st_size / 1024
        print(f"✅ Exported ({size_kb:.2f} KB)")
    else:
        raise IOError(f"Failed to export model")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export model from MLflow")
    parser.add_argument("--metric", default="accuracy", help="Metric for selection")
    parser.add_argument("--run-id", help="Specific run ID to export")
    parser.add_argument("--output", default="/mnt/artifacts/model.pkl", help="Output path")
    parser.add_argument("--experiment", help="Experiment name pattern")

    args = parser.parse_args()

    try:
        if args.run_id:
            run_id = args.run_id
        else:
            run_id = find_best_run(args.metric, args.experiment)

        output_path = export_model(run_id, args.output)

        print(f"\n{'='*60}")
        print(f"SUCCESS")
        print(f"{'='*60}")
        print(f"Model: {output_path}")
        print(f"\nNext: Register training set for monitoring")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
