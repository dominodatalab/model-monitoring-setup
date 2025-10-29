#!/usr/bin/env python3
"""
Register Training Set with Domino Model Monitor

Register a training dataset from a file or DataFrame for baseline metrics.

Usage:
    python register_training_set.py --file data/training.csv
    python register_training_set.py --file data/training.parquet --name "My Training Set"
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd

try:
    from domino.training_sets import TrainingSetClient, model
    DOMINO_AVAILABLE = True
except ImportError:
    DOMINO_AVAILABLE = False
    print("⚠️  Warning: domino.training_sets not available")
    print("Installing dominodatalab package...")
    
    import subprocess
    try:
        # Install with --user and ignore dependency conflicts
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--user", 
            "--no-deps", "dominodatalab"
        ])
        print("✅ dominodatalab package installed successfully")
        
        # Restart Python path to pick up user-installed packages
        import site
        site.main()
        
        # Try importing again after installation
        from domino.training_sets import TrainingSetClient, model
        DOMINO_AVAILABLE = True
        print("✅ domino.training_sets now available")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dominodatalab: {e}")
        print("Try running manually: pip install --user dominodatalab")
    except ImportError as e:
        print(f"❌ Installation succeeded but import still failed: {e}")
        print("The package may need to be installed in the Domino environment.")
        print("Try running manually: pip install --user dominodatalab")


def register_from_file(file_path: str, name: str = None, key_columns: list = None):
    """
    Register a training set from a file.

    Args:
        file_path: Path to CSV, Parquet, or other pandas-readable file
        name: Optional name for the training set
        key_columns: Optional list of key columns for tracking
    """
    if not DOMINO_AVAILABLE:
        print("❌ Cannot register training set - Domino SDK not available")
        sys.exit(1)

    file_path = Path(file_path)

    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    # Load the data
    print(f"📂 Loading data from: {file_path}")

    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
    elif file_path.suffix in ['.parquet', '.pq']:
        df = pd.read_parquet(file_path)
    else:
        # Try pandas read_csv as default
        df = pd.read_csv(file_path)

    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")

    # Get username from environment
    username = os.environ.get("DOMINO_STARTING_USERNAME", "user")
    
    # Prompt for name if not provided
    if name is None:
        print(f"\n📝 Training set name required (must be unique across Domino instance)")
        suggested_name = f"training_set_{file_path.stem}"
        name = input(f"Enter training set name [{suggested_name}]: ").strip()
        if not name:
            name = suggested_name
    
    # Clean the name to be valid (alphanumeric and underscores only)
    import re
    original_name = name
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)  # Replace invalid chars with underscores
    name = re.sub(r'_+', '_', name)  # Replace multiple underscores with single
    name = name.strip('_')  # Remove leading/trailing underscores
    
    # Add username suffix to ensure uniqueness
    if not name.endswith(f"_{username}"):
        name = f"{name}_{username}"

    # Show name cleaning if it changed
    if name != original_name:
        print(f"\n🔧 Name cleaned: '{original_name}' → '{name}'")

    # Register the training set
    print(f"\n📤 Registering training set: {name}")
    
    # Identify target column
    # Look for common target column names, otherwise use last column
    target_indicators = ['target', 'label', 'y', 'output', 'prediction', 'price', 'value', 'score']
    target_columns = []
    
    for col in df.columns:
        if any(indicator in col.lower() for indicator in target_indicators):
            target_columns = [col]
            break
    
    if not target_columns:
        # Default to last column if no clear target found
        target_columns = [df.columns[-1]]
    
    # Auto-detect likely model type based on target column
    if target_columns:
        target_col = target_columns[0]
        target_dtype = df[target_col].dtype
        unique_values = df[target_col].nunique()
        
        if target_dtype == 'object' or unique_values <= 10:
            suggested_type = "classification"
        else:
            suggested_type = "regression"
            
        print(f"\n🎯 Target column '{target_col}' detected:")
        print(f"   - Data type: {target_dtype}")
        print(f"   - Unique values: {unique_values}")
        print(f"   - Sample values: {list(df[target_col].unique()[:5])}")
    else:
        suggested_type = "classification"
        print(f"\n⚠️  No clear target column detected")
    
    # Prompt user to confirm model type
    print(f"\n📊 Model type selection:")
    print(f"1. Classification (predicts categories/classes)")
    print(f"2. Regression (predicts continuous numeric values)")
    
    while True:
        choice = input(f"Select model type [1 for classification, 2 for regression] (suggested: {suggested_type}): ").strip()
        if choice == "1" or (choice == "" and suggested_type == "classification"):
            model_type = "classification"
            break
        elif choice == "2" or (choice == "" and suggested_type == "regression"):
            model_type = "regression"
            break
        else:
            print("Please enter 1 for classification or 2 for regression")
    
    # Analyze column types based on model type
    categorical_columns = []
    ordinal_columns = []
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # For regression, don't include numeric targets in categorical
            # For classification, include all object dtype columns (including target)
            if model_type == "regression" and col in target_columns:
                print(f"   ⚠️  Warning: Target column '{col}' is text but model type is regression")
                print(f"      Consider converting to numeric or changing to classification")
            categorical_columns.append(col)
    
    # For regression, ensure target is not in categorical columns if it's numeric
    if model_type == "regression" and target_columns:
        target_col = target_columns[0]
        if target_col in categorical_columns and df[target_col].dtype != 'object':
            categorical_columns.remove(target_col)
    
    numeric_features = [col for col in df.columns if col not in target_columns and col not in categorical_columns]
    
    print(f"\n   📋 Schema analysis ({model_type} model):")
    print(f"   - Target columns: {target_columns}")
    print(f"   - Categorical columns: {categorical_columns}")
    print(f"   - Numeric feature columns: {numeric_features}")
    
    # Set up comprehensive monitoring metadata for model monitoring
    monitoring_meta = model.MonitoringMeta(
        categorical_columns=categorical_columns,
        timestamp_columns=[],  # No timestamp columns by default
        ordinal_columns=ordinal_columns  # No ordinal columns by default
    )
    
    # For monitoring, we only want features (not target) in the training set
    # Remove target column from the dataframe used for training set registration
    feature_df = df.drop(columns=target_columns) if target_columns else df
    
    # Update monitoring_meta to exclude target from categorical columns
    feature_categorical_columns = [col for col in categorical_columns if col not in target_columns]
    monitoring_meta_features = model.MonitoringMeta(
        categorical_columns=feature_categorical_columns,
        timestamp_columns=[],
        ordinal_columns=ordinal_columns
    )
    
    print(f"\n   📋 Final training set schema (features only for monitoring):")
    print(f"   - Feature columns: {list(feature_df.columns)}")
    print(f"   - Categorical features: {feature_categorical_columns}")
    
    ts_version = TrainingSetClient.create_training_set_version(
        training_set_name=name,
        df=feature_df,  # Only features, no target
        key_columns=key_columns or [],
        target_columns=[],  # No target columns for monitoring
        monitoring_meta=monitoring_meta_features
    )

    print(f"✅ Training set registered successfully")
    print(f"   Name: {name}")
    print(f"   Path: {ts_version.path}")

    return ts_version


def register_from_dataframe(df: pd.DataFrame, name: str, key_columns: list = None):
    """
    Register a training set from a DataFrame.

    Args:
        df: pandas DataFrame
        name: Name for the training set
        key_columns: Optional list of key columns for tracking
    """
    if not DOMINO_AVAILABLE:
        print("❌ Cannot register training set - Domino SDK not available")
        sys.exit(1)

    print(f"📊 Registering DataFrame as training set")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")

    training_set = model.TrainingSet(
        training_set_name=name,
        training_data=df,
        key_columns=key_columns or []
    )

    ts_client = client.TrainingSetClient()
    ts_version = ts_client.put(training_set)

    print(f"✅ Training set registered successfully")
    print(f"   Name: {ts_version.training_set_name}")
    print(f"   Version: {ts_version.training_set_version}")

    return ts_version


def main():
    parser = argparse.ArgumentParser(
        description="Register a training set with Domino Model Monitor"
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to training data file (CSV, Parquet, etc.)"
    )
    parser.add_argument(
        "--name",
        help="Name for the training set (auto-generated if not provided)"
    )
    parser.add_argument(
        "--key-columns",
        nargs="+",
        help="Key columns for tracking (optional)"
    )

    args = parser.parse_args()

    try:
        register_from_file(args.file, args.name, args.key_columns)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    print("\n" + "="*60)
    print("✅ TRAINING SET REGISTERED SUCCESSFULLY")
    print("="*60)
    print("\nNext steps:")
    print("2. Create Model Deployment Script:")
    print("   - Add prediction capture to your model script (see prediction_capture_example.py)")
    print("   - Deploy as Domino Model API endpoint")
    print("   - Select this training set in Model Monitor UI (Settings > Monitoring)")
    print("\n3. Configure data sources in three places (see README Step 4)")
    print("\n3. Configure monitoring:")
    print("   python 3_setup_monitoring.py")
    print("="*60)


if __name__ == "__main__":
    main()
