#!/usr/bin/env python3
"""
Generate Drifted Input Data

This script generates input data with configurable drift introduced.
Useful for testing model monitoring drift detection capabilities.

Usage:
    python generate_drifted_data.py --drift-level low --count 100
    python generate_drifted_data.py --drift-level medium --count 50 --output drifted_data.csv
    python generate_drifted_data.py --drift-level high --count 200
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


class DriftedDataGenerator:
    """Generate test data with controlled drift from training distribution."""

    def __init__(self, training_data_path: str = '/mnt/artifacts/training_data.csv'):
        """
        Initialize generator with training data statistics.

        Args:
            training_data_path: Path to training data CSV
        """
        self.training_data = pd.read_csv(training_data_path)

        # Extract feature names (exclude target)
        self.feature_names = [col for col in self.training_data.columns if col != 'target']

        # Calculate statistics from training data
        self.means = self.training_data[self.feature_names].mean()
        self.stds = self.training_data[self.feature_names].std()
        self.mins = self.training_data[self.feature_names].min()
        self.maxs = self.training_data[self.feature_names].max()

        print(f"📊 Loaded training data: {len(self.training_data)} samples")
        print(f"   Features: {len(self.feature_names)}")
        print(f"   Mean range: [{self.means.min():.3f}, {self.means.max():.3f}]")
        print(f"   Std range: [{self.stds.min():.3f}, {self.stds.max():.3f}]")

    def generate_drifted_data(
        self,
        n_samples: int = 100,
        drift_level: str = 'medium',
        drift_type: str = 'mean_shift',
        random_seed: int = 42
    ) -> pd.DataFrame:
        """
        Generate data with drift.

        Args:
            n_samples: Number of samples to generate
            drift_level: 'low', 'medium', or 'high' drift intensity
            drift_type: Type of drift to introduce:
                - 'mean_shift': Shift feature means
                - 'variance_change': Change feature variance
                - 'mixed': Combination of mean shift and variance change
                - 'covariate': Shift in feature correlations
            random_seed: Random seed for reproducibility

        Returns:
            DataFrame with drifted data
        """
        np.random.seed(random_seed)

        # Define drift parameters
        drift_params = {
            'low': {'mean_shift': 0.3, 'variance_mult': 1.2, 'affected_features': 0.3},
            'medium': {'mean_shift': 0.8, 'variance_mult': 1.5, 'affected_features': 0.5},
            'high': {'mean_shift': 1.5, 'variance_mult': 2.0, 'affected_features': 0.8}
        }

        params = drift_params[drift_level]
        print(f"\n🔄 Generating {n_samples} samples with {drift_level} {drift_type} drift")
        print(f"   Mean shift: {params['mean_shift']} std devs")
        print(f"   Variance multiplier: {params['variance_mult']}x")
        print(f"   Affected features: {params['affected_features']*100:.0f}%")

        # Determine which features to affect
        n_affected = int(len(self.feature_names) * params['affected_features'])
        affected_features = np.random.choice(self.feature_names, n_affected, replace=False)

        print(f"   Drifting features: {list(affected_features)}")

        # Generate base data from training distribution
        data = {}

        for feature in self.feature_names:
            # Generate from normal distribution
            base_mean = self.means[feature]
            base_std = self.stds[feature]

            if feature in affected_features:
                if drift_type == 'mean_shift' or drift_type == 'mixed':
                    # Shift the mean
                    drift_direction = np.random.choice([-1, 1])
                    new_mean = base_mean + (drift_direction * params['mean_shift'] * base_std)
                else:
                    new_mean = base_mean

                if drift_type == 'variance_change' or drift_type == 'mixed':
                    # Change the variance
                    new_std = base_std * params['variance_mult']
                else:
                    new_std = base_std

                # Generate drifted samples
                samples = np.random.normal(new_mean, new_std, n_samples)

            else:
                # Keep original distribution for unaffected features
                samples = np.random.normal(base_mean, base_std, n_samples)

            # Clip to reasonable range (3 std devs from training mean)
            samples = np.clip(samples, self.mins[feature] - base_std, self.maxs[feature] + base_std)

            data[feature] = samples

        df = pd.DataFrame(data)

        # Add metadata columns
        df['drift_level'] = drift_level
        df['drift_type'] = drift_type
        df['generated_at'] = datetime.now().isoformat()

        return df

    def compare_distributions(self, drifted_data: pd.DataFrame):
        """
        Print comparison between training and drifted data distributions.

        Args:
            drifted_data: Generated drifted data
        """
        print("\n📈 Distribution Comparison:")
        print(f"{'Feature':<12} {'Train Mean':<12} {'Drift Mean':<12} {'Δ Mean':<10} {'Train Std':<12} {'Drift Std':<12} {'Δ Std':<10}")
        print("-" * 90)

        for feature in self.feature_names:
            train_mean = self.means[feature]
            train_std = self.stds[feature]
            drift_mean = drifted_data[feature].mean()
            drift_std = drifted_data[feature].std()

            delta_mean = drift_mean - train_mean
            delta_std = drift_std - train_std

            # Highlight if change is significant
            marker = "🔴" if abs(delta_mean / train_std) > 0.5 else ""

            print(f"{feature:<12} {train_mean:>11.3f} {drift_mean:>11.3f} {delta_mean:>9.3f} {train_std:>11.3f} {drift_std:>11.3f} {delta_std:>9.3f} {marker}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate test data with drift for model monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate low drift data
  python generate_drifted_data.py --drift-level low --count 100

  # Generate high mean shift drift
  python generate_drifted_data.py --drift-level high --drift-type mean_shift --count 200

  # Generate medium mixed drift with custom output
  python generate_drifted_data.py --drift-level medium --drift-type mixed --output test_drift.csv
        """
    )

    parser.add_argument(
        '--drift-level',
        type=str,
        choices=['low', 'medium', 'high'],
        default='medium',
        help="Intensity of drift to introduce (default: medium)"
    )

    parser.add_argument(
        '--drift-type',
        type=str,
        choices=['mean_shift', 'variance_change', 'mixed', 'covariate'],
        default='mean_shift',
        help="Type of drift to introduce (default: mean_shift)"
    )

    parser.add_argument(
        '--count',
        type=int,
        default=100,
        help="Number of samples to generate (default: 100)"
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help="Output CSV file path (default: drifted_data_<level>_<timestamp>.csv)"
    )

    parser.add_argument(
        '--training-data',
        type=str,
        default='/mnt/artifacts/training_data.csv',
        help="Path to training data CSV (default: /mnt/artifacts/training_data.csv)"
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("🌊 Drifted Data Generator")
    print("=" * 70)

    # Initialize generator
    try:
        generator = DriftedDataGenerator(args.training_data)
    except FileNotFoundError:
        print(f"❌ Error: Training data not found at {args.training_data}")
        return 1

    # Generate drifted data
    drifted_data = generator.generate_drifted_data(
        n_samples=args.count,
        drift_level=args.drift_level,
        drift_type=args.drift_type,
        random_seed=args.seed
    )

    # Show distribution comparison
    generator.compare_distributions(drifted_data)

    # Save to file
    if args.output:
        output_path = args.output
        # If output path doesn't include a directory, save to /mnt/artifacts
        if '/' not in output_path:
            output_path = f"/mnt/artifacts/{output_path}"
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"/mnt/artifacts/drifted_data_{args.drift_level}_{timestamp}.csv"

    output_file = Path(output_path)
    # Ensure parent directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    drifted_data.to_csv(output_file, index=False)

    print(f"\n✅ Saved {len(drifted_data)} drifted samples to: {output_file}")
    print(f"   File size: {output_file.stat().st_size / 1024:.1f} KB")

    # Show sample rows
    print(f"\n📋 Sample rows (first 3):")
    print(drifted_data[generator.feature_names].head(3).to_string(index=False))

    print("\n" + "=" * 70)
    print("✅ Drift generation complete!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
