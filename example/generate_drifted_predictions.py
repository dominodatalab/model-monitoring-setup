#!/usr/bin/env python3
"""
Generate Drifted Predictions with Balanced Classes

This script generates 50 predictions from test data with different types of drift
applied to each prediction. Ensures equal distribution across 4 classes and 
applies varying drift patterns.

Usage:
    python generate_drifted_predictions.py
    python generate_drifted_predictions.py --output drifted_predictions.csv
    python generate_drifted_predictions.py --seed 123
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import random
import os
import requests
import time
import sys
from typing import List, Dict, Any

# Add parent directory to path to import config_loader
sys.path.insert(0, str(Path(__file__).parent.parent))


class DriftedPredictionGenerator:
    """Generate drifted predictions with balanced class distribution."""

    def __init__(self, test_data_path: str = '/mnt/artifacts/test_data.csv'):
        """
        Initialize generator with test data.

        Args:
            test_data_path: Path to test data CSV
        """
        self.test_data = pd.read_csv(test_data_path)
        
        # Extract feature names (exclude target)
        self.feature_names = [col for col in self.test_data.columns if col != 'target']
        
        # Get unique classes and their counts
        self.classes = sorted(self.test_data['target'].unique())
        self.class_counts = self.test_data['target'].value_counts().sort_index()
        
        # Calculate feature statistics for drift generation
        self.feature_stats = {}
        for feature in self.feature_names:
            self.feature_stats[feature] = {
                'mean': self.test_data[feature].mean(),
                'std': self.test_data[feature].std(),
                'min': self.test_data[feature].min(),
                'max': self.test_data[feature].max()
            }
        
        print(f"📊 Loaded test data: {len(self.test_data)} samples")
        print(f"   Features: {len(self.feature_names)}")
        print(f"   Classes: {self.classes}")
        print(f"   Class distribution: {dict(self.class_counts)}")

    def get_drift_patterns(self) -> List[Dict[str, Any]]:
        """
        Define different drift patterns to apply.
        
        Returns:
            List of drift pattern configurations
        """
        patterns = [
            # Light drift patterns
            {
                'name': 'light_mean_shift',
                'description': 'Light mean shift in 2-3 features',
                'features_affected': 0.3,
                'mean_shift_factor': 0.2,
                'variance_factor': 1.0,
                'noise_factor': 1.0
            },
            {
                'name': 'light_variance_increase',
                'description': 'Light variance increase in 2-3 features', 
                'features_affected': 0.3,
                'mean_shift_factor': 0.0,
                'variance_factor': 1.3,
                'noise_factor': 1.1
            },
            # Medium drift patterns  
            {
                'name': 'medium_mean_shift',
                'description': 'Medium mean shift in 4-5 features',
                'features_affected': 0.5,
                'mean_shift_factor': 0.5,
                'variance_factor': 1.0,
                'noise_factor': 1.0
            },
            {
                'name': 'medium_mixed_drift',
                'description': 'Medium mixed drift (mean + variance)',
                'features_affected': 0.4,
                'mean_shift_factor': 0.4,
                'variance_factor': 1.4,
                'noise_factor': 1.2
            },
            # Heavy drift patterns
            {
                'name': 'heavy_mean_shift',
                'description': 'Heavy mean shift in 5-7 features',
                'features_affected': 0.7,
                'mean_shift_factor': 0.8,
                'variance_factor': 1.0,
                'noise_factor': 1.0
            },
            {
                'name': 'heavy_variance_change',
                'description': 'Heavy variance change in 4-6 features',
                'features_affected': 0.6,
                'mean_shift_factor': 0.1,
                'variance_factor': 2.0,
                'noise_factor': 1.5
            },
            {
                'name': 'extreme_mixed_drift',
                'description': 'Extreme mixed drift across features',
                'features_affected': 0.8,
                'mean_shift_factor': 1.0,
                'variance_factor': 1.8,
                'noise_factor': 1.3
            },
            # Subtle/complex patterns
            {
                'name': 'correlation_drift',
                'description': 'Correlation structure changes',
                'features_affected': 0.5,
                'mean_shift_factor': 0.3,
                'variance_factor': 1.2,
                'noise_factor': 1.4
            },
            {
                'name': 'periodic_drift',
                'description': 'Periodic oscillation in features',
                'features_affected': 0.4,
                'mean_shift_factor': 0.6,
                'variance_factor': 1.1,
                'noise_factor': 1.0
            },
            {
                'name': 'gradual_drift',
                'description': 'Gradual drift accumulation',
                'features_affected': 0.6,
                'mean_shift_factor': 0.7,
                'variance_factor': 1.3,
                'noise_factor': 1.1
            }
        ]
        return patterns

    def apply_drift_to_sample(self, sample: pd.Series, pattern: Dict[str, Any], 
                             prediction_index: int) -> pd.Series:
        """
        Apply drift pattern to a single sample.
        
        Args:
            sample: Original data sample
            pattern: Drift pattern configuration
            prediction_index: Index for time-varying drift
            
        Returns:
            Drifted sample
        """
        drifted_sample = sample.copy()
        
        # Select features to affect
        n_affected = max(1, int(len(self.feature_names) * pattern['features_affected']))
        affected_features = np.random.choice(self.feature_names, n_affected, replace=False)
        
        for feature in affected_features:
            original_value = sample[feature]
            stats = self.feature_stats[feature]
            
            # Apply mean shift
            if pattern['mean_shift_factor'] > 0:
                # Random direction for mean shift
                direction = np.random.choice([-1, 1])
                mean_shift = direction * pattern['mean_shift_factor'] * stats['std']
                
                # For periodic drift, add sine wave component
                if 'periodic' in pattern['name']:
                    phase = (prediction_index / 10) * 2 * np.pi
                    mean_shift *= (1 + 0.3 * np.sin(phase))
                
                # For gradual drift, scale by prediction index
                elif 'gradual' in pattern['name']:
                    mean_shift *= (1 + prediction_index * 0.02)
                
                drifted_sample[feature] += mean_shift
            
            # Apply variance change
            if pattern['variance_factor'] != 1.0:
                # Center around current value and scale variance
                centered_value = drifted_sample[feature] - stats['mean']
                scaled_value = centered_value * pattern['variance_factor']
                drifted_sample[feature] = stats['mean'] + scaled_value
            
            # Add additional noise
            if pattern['noise_factor'] > 1.0:
                noise_std = stats['std'] * (pattern['noise_factor'] - 1.0) * 0.1
                noise = np.random.normal(0, noise_std)
                drifted_sample[feature] += noise
            
            # Clip to reasonable range (within 4 std devs of original mean)
            min_val = stats['mean'] - 4 * stats['std']
            max_val = stats['mean'] + 4 * stats['std']
            drifted_sample[feature] = np.clip(drifted_sample[feature], min_val, max_val)
        
        return drifted_sample

    def generate_balanced_predictions(self, n_predictions: int = 50, 
                                    random_seed: int = 42) -> pd.DataFrame:
        """
        Generate predictions with balanced class distribution and varied drift.
        
        Args:
            n_predictions: Total number of predictions to generate
            random_seed: Random seed for reproducibility
            
        Returns:
            DataFrame with drifted predictions and metadata
        """
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        print(f"\n🎯 Generating {n_predictions} balanced predictions with drift")
        
        # Calculate samples per class (ensure balanced distribution)
        samples_per_class = n_predictions // len(self.classes)
        remainder = n_predictions % len(self.classes)
        
        class_targets = []
        for i, class_name in enumerate(self.classes):
            count = samples_per_class + (1 if i < remainder else 0)
            class_targets.extend([class_name] * count)
        
        # Shuffle to randomize order
        random.shuffle(class_targets)
        
        print(f"   Class distribution: {dict(pd.Series(class_targets).value_counts().sort_index())}")
        
        # Get drift patterns
        drift_patterns = self.get_drift_patterns()
        print(f"   Available drift patterns: {len(drift_patterns)}")
        
        predictions = []
        
        for i in range(n_predictions):
            target_class = class_targets[i]
            
            # Select a random sample from the target class
            class_samples = self.test_data[self.test_data['target'] == target_class]
            base_sample = class_samples.sample(n=1).iloc[0]
            
            # Select drift pattern (cycle through patterns with some randomness)
            pattern = drift_patterns[i % len(drift_patterns)]
            
            # Add some randomness to pattern selection
            if random.random() < 0.3:  # 30% chance to use a different random pattern
                pattern = random.choice(drift_patterns)
            
            # Apply drift to features
            drifted_sample = self.apply_drift_to_sample(base_sample, pattern, i)
            
            # Create prediction record
            prediction = {
                'prediction_id': f'pred_{i+1:03d}',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'actual_target': target_class,
                'drift_pattern': pattern['name'],
                'drift_description': pattern['description'],
                'prediction_index': i
            }
            
            # Add feature values
            for feature in self.feature_names:
                prediction[f'feature_{feature}'] = drifted_sample[feature]
            
            predictions.append(prediction)
            
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i + 1}/{n_predictions}")
        
        df_predictions = pd.DataFrame(predictions)
        
        # Add summary statistics
        print(f"\n📊 Generation Summary:")
        print(f"   Total predictions: {len(df_predictions)}")
        print(f"   Class distribution:")
        class_dist = df_predictions['actual_target'].value_counts().sort_index()
        for class_name, count in class_dist.items():
            print(f"     {class_name}: {count}")
        
        print(f"   Drift patterns used:")
        pattern_dist = df_predictions['drift_pattern'].value_counts()
        for pattern, count in pattern_dist.items():
            print(f"     {pattern}: {count}")
        
        return df_predictions

    def compare_original_vs_drifted(self, predictions_df: pd.DataFrame):
        """
        Compare original test data statistics with drifted predictions.
        
        Args:
            predictions_df: Generated predictions with drift
        """
        print(f"\n📈 Drift Analysis:")
        print(f"{'Feature':<12} {'Original Mean':<13} {'Drifted Mean':<13} {'Δ Mean':<10} {'Original Std':<12} {'Drifted Std':<12} {'Δ Std':<8}")
        print("-" * 90)
        
        for feature in self.feature_names:
            feature_col = f'feature_{feature}'
            
            if feature_col in predictions_df.columns:
                orig_mean = self.feature_stats[feature]['mean']
                orig_std = self.feature_stats[feature]['std']
                drift_mean = predictions_df[feature_col].mean()
                drift_std = predictions_df[feature_col].std()
                
                delta_mean = drift_mean - orig_mean
                delta_std = drift_std - orig_std
                
                # Highlight significant changes
                marker = "🔴" if abs(delta_mean / orig_std) > 0.5 else ""
                
                print(f"{feature:<12} {orig_mean:>12.3f} {drift_mean:>12.3f} {delta_mean:>9.3f} {orig_std:>11.3f} {drift_std:>11.3f} {delta_std:>7.3f} {marker}")


def call_model_api_with_predictions(predictions_df: pd.DataFrame) -> bool:
    """
    Call model API with generated predictions (based on 4_generate_predictions.py).
    
    Args:
        predictions_df: DataFrame with generated predictions
        
    Returns:
        True if all API calls succeeded, False otherwise
    """
    try:
        from config_loader import get_config
        config = get_config()
        api_url = config.model_api_url
        api_token = config.model_api_token
    except Exception as e:
        print(f"❌ Could not load config: {e}")
        return False
    
    if not api_url or not api_token:
        print("❌ Model API URL or token not configured")
        return False
    
    success_count = 0
    total_count = len(predictions_df)
    
    print(f"🚀 Making {total_count} API calls to: {api_url}")
    
    for idx, row in predictions_df.iterrows():
        try:
            # Prepare payload with feature data (similar to 4_generate_predictions.py)
            feature_data = {}
            for col in predictions_df.columns:
                if col.startswith('feature_'):
                    # Remove 'feature_' prefix for API call
                    feature_name = col.replace('feature_', '')
                    feature_data[feature_name] = float(row[col])
            
            payload = {'data': feature_data}
            
            # Make API request with HTTP Basic Auth (like 4_generate_predictions.py)
            response = requests.post(
                api_url,
                json=payload,
                auth=(api_token, api_token),
                timeout=60
            )
            
            response.raise_for_status()
            result = response.json()
            
            success_count += 1
            if (idx + 1) % 10 == 0:
                print(f"   Progress: {idx + 1}/{total_count}")
            
            # Add delay like in original script
            time.sleep(random.uniform(1, 3))
            
        except Exception as e:
            print(f"   ❌ API call {idx + 1} failed: {e}")
    
    print(f"✅ API calls completed: {success_count}/{total_count} successful")
    return success_count == total_count


def upload_ground_truth(predictions_df: pd.DataFrame) -> bool:
    """
    Upload ground truth to data source (based on 4_generate_predictions.py).
    
    Args:
        predictions_df: DataFrame with generated predictions
        
    Returns:
        True if upload succeeded, False otherwise
    """
    try:
        from domino.data_sources import DataSourceClient
        from config_loader import get_config
        import csv
        import io
        
        config = get_config()
        datasource_name = config.ground_truth_datasource
        model_id = config.model_monitor_id
        
    except ImportError:
        print("⚠️ DataSourceClient not available")
        return False
    except Exception as e:
        print(f"❌ Could not load config: {e}")
        return False
    
    try:
        print(f"📤 Uploading ground truth to: {datasource_name}")
        
        # Create CSV records (like 4_generate_predictions.py)
        records = []
        for _, row in predictions_df.iterrows():
            records.append({
                'event_id': row['prediction_id'],
                'actual_target': row['actual_target'],
                'timestamp': row['timestamp']
            })
        
        # Create CSV buffer
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=['event_id', 'actual_target', 'timestamp'])
        writer.writeheader()
        writer.writerows(records)
        csv_bytes = csv_buffer.getvalue().encode('utf-8')
        
        # Upload to data source
        datasource = DataSourceClient().get_datasource(datasource_name)
        today = datetime.now().strftime('%Y-%m-%d')
        s3_key = f"ground_truth/{model_id}/{today}_drifted.csv"
        
        bytes_buffer = io.BytesIO(csv_bytes)
        datasource.upload_fileobj(s3_key, bytes_buffer)
        
        print(f"✅ Ground truth uploaded: {s3_key}")
        print(f"   Records: {len(records)}")
        return True
        
    except Exception as e:
        print(f"❌ Ground truth upload failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate drifted predictions with balanced classes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 50 predictions with default settings
  python generate_drifted_predictions.py

  # Generate with custom output file
  python generate_drifted_predictions.py --output my_predictions.csv

  # Generate with custom seed for reproducibility
  python generate_drifted_predictions.py --seed 123
        """
    )

    parser.add_argument(
        '--count',
        type=int,
        default=50,
        help="Number of predictions to generate (default: 50)"
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help="Output CSV file path (default: drifted_predictions_<timestamp>.csv)"
    )

    parser.add_argument(
        '--test-data',
        type=str,
        default='/mnt/artifacts/test_data.csv',
        help="Path to test data CSV (default: /mnt/artifacts/test_data.csv)"
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("🎯 Drifted Prediction Generator")
    print("=" * 70)

    # Initialize generator
    try:
        generator = DriftedPredictionGenerator(args.test_data)
    except FileNotFoundError:
        print(f"❌ Error: Test data not found at {args.test_data}")
        return 1

    # Generate drifted predictions
    predictions = generator.generate_balanced_predictions(
        n_predictions=args.count,
        random_seed=args.seed
    )

    # Show drift analysis
    generator.compare_original_vs_drifted(predictions)

    # Save to file
    if args.output:
        output_path = args.output
        # If output path doesn't include a directory, save to /mnt/artifacts
        if '/' not in output_path:
            output_path = f"/mnt/artifacts/{output_path}"
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"/mnt/artifacts/drifted_predictions_{timestamp}.csv"

    output_file = Path(output_path)
    # Ensure parent directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_file, index=False)

    print(f"\n✅ Saved {len(predictions)} drifted predictions to: {output_file}")
    print(f"   File size: {output_file.stat().st_size / 1024:.1f} KB")

    # Show sample predictions
    print(f"\n📋 Sample predictions (first 3):")
    sample_cols = ['prediction_id', 'actual_target', 'drift_pattern'] + [f'feature_{f}' for f in generator.feature_names[:3]]
    print(predictions[sample_cols].head(3).to_string(index=False))

    # Call model API with generated data
    print("\n" + "=" * 70)
    print("🔗 CALLING MODEL API")
    print("=" * 70)
    
    api_success = call_model_api_with_predictions(predictions)
    
    # Upload ground truth if API calls were successful
    gt_success = False
    if api_success:
        print("\n" + "=" * 70)
        print("📤 UPLOADING GROUND TRUTH")
        print("=" * 70)
        gt_success = upload_ground_truth(predictions)
    
    # Clean up the CSV file
    try:
        output_file.unlink()
        print(f"\n🗑️ Cleaned up temporary file: {output_file}")
    except Exception as e:
        print(f"⚠️ Could not remove temporary file {output_file}: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"Predictions generated: {len(predictions)}")
    print(f"API calls successful: {'Yes' if api_success else 'No'}")
    print(f"Ground truth uploaded: {'Yes' if gt_success else 'No'}")
    
    if gt_success:
        print(f"\n⏰ Wait 1 hour, then run:")
        print(f"   python 5_upload_ground_truth.py")
    
    print("\n" + "=" * 70)
    print("✅ Drift prediction generation complete!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())