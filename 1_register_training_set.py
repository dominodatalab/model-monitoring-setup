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

    # Auto-generate name if not provided
    if name is None:
        name = f"training_set_{file_path.stem}"

    # Register the training set
    print(f"\n📤 Registering training set: {name}")
    
    # Identify target column (assuming last column is target)
    target_columns = [df.columns[-1]] if 'target' in df.columns[-1].lower() else []
    
    # Set up monitoring metadata for model monitoring
    monitoring_meta = model.MonitoringMeta(
        categorical_columns=[col for col in df.columns if df[col].dtype == 'object' and col not in target_columns],
        timestamp_columns=[],
    )
    
    ts_version = TrainingSetClient.create_training_set_version(
        training_set_name=name,
        df=df,
        key_columns=key_columns or [],
        target_columns=target_columns,
        monitoring_meta=monitoring_meta
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
