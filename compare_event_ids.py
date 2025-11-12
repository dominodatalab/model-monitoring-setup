#!/usr/bin/env python3
"""
Compare event IDs between recent prediction parquet files and ground truth CSV
"""

import pandas as pd
from pathlib import Path
from domino.data_sources import DataSourceClient
import io

# Read the most recent parquet file
print("=" * 80)
print("READING PREDICTION DATA")
print("=" * 80)

# Most recent parquet file
recent_parquet = "/mnt/data/prediction_data/690eb04f8a0ee66d0ee2394e/$$date$$=2025-11-12Z/$$hour$$=12Z/predictions_48afe7d5-4765-49cd-8f14-a2301193f0c1.parquet"
print(f"\nReading: {recent_parquet}")

predictions_df = pd.read_parquet(recent_parquet)
print(f"\nPredictions DataFrame shape: {predictions_df.shape}")
print(f"Columns: {list(predictions_df.columns)}")

# Check if event_id column exists
if 'event_id' in predictions_df.columns:
    prediction_event_ids = set(predictions_df['event_id'].dropna())
    print(f"\nUnique event IDs in predictions: {len(prediction_event_ids)}")
    print(f"Sample event IDs: {list(prediction_event_ids)[:5]}")
else:
    print("\n⚠️ No 'event_id' column found in predictions!")
    print(f"Available columns: {list(predictions_df.columns)}")
    prediction_event_ids = set()

# Access ground truth from data source
print("\n" + "=" * 80)
print("READING GROUND TRUTH DATA")
print("=" * 80)

try:
    # Instantiate client and fetch datasource
    object_store = DataSourceClient().get_datasource("model-monitor-storage")
    print("\n✅ Connected to model-monitor-storage data source")

    # List objects to verify the file exists
    print("\nListing objects in ground_truth/690eb116fcc676c364a704ef/...")
    objects = object_store.list_objects(prefix="ground_truth/690eb116fcc676c364a704ef/")

    # Objects are returned as _Object instances with .key attribute
    object_keys = []
    for obj in objects:
        if hasattr(obj, 'key'):
            object_keys.append(obj.key)
        else:
            object_keys.append(str(obj))

    matching_files = [key for key in object_keys if key.endswith('.csv')]
    print(f"Found {len(matching_files)} CSV files:")
    for obj in matching_files[:10]:  # Show first 10
        print(f"  - {obj}")

    # Download the specific ground truth file
    gt_key = "ground_truth/690eb116fcc676c364a704ef/2025-11-12.csv"
    print(f"\nDownloading: {gt_key}")

    # Download to file object
    f = io.BytesIO()
    object_store.download_fileobj(gt_key, f)
    f.seek(0)

    # Read CSV
    ground_truth_df = pd.read_csv(f)
    print(f"\nGround Truth DataFrame shape: {ground_truth_df.shape}")
    print(f"Columns: {list(ground_truth_df.columns)}")

    # Check for event_id column
    if 'event_id' in ground_truth_df.columns:
        gt_event_ids = set(ground_truth_df['event_id'].dropna())
        print(f"\nUnique event IDs in ground truth: {len(gt_event_ids)}")
        print(f"Sample event IDs: {list(gt_event_ids)[:5]}")
    else:
        print("\n⚠️ No 'event_id' column found in ground truth!")
        print(f"Available columns: {list(ground_truth_df.columns)}")
        gt_event_ids = set()

    # Compare event IDs
    if prediction_event_ids and gt_event_ids:
        print("\n" + "=" * 80)
        print("COMPARISON RESULTS")
        print("=" * 80)

        # Find matches and differences
        matching_ids = prediction_event_ids & gt_event_ids
        only_in_predictions = prediction_event_ids - gt_event_ids
        only_in_ground_truth = gt_event_ids - prediction_event_ids

        print(f"\n📊 Summary:")
        print(f"  - Total event IDs in predictions: {len(prediction_event_ids)}")
        print(f"  - Total event IDs in ground truth: {len(gt_event_ids)}")
        print(f"  - Matching event IDs: {len(matching_ids)}")
        print(f"  - Only in predictions: {len(only_in_predictions)}")
        print(f"  - Only in ground truth: {len(only_in_ground_truth)}")

        if matching_ids:
            print(f"\n✅ Found {len(matching_ids)} matching event IDs")
            print(f"Sample matches: {list(matching_ids)[:5]}")

        if only_in_predictions:
            print(f"\n⚠️ {len(only_in_predictions)} event IDs only in predictions")
            print(f"Sample: {list(only_in_predictions)[:5]}")

        if only_in_ground_truth:
            print(f"\n⚠️ {len(only_in_ground_truth)} event IDs only in ground truth")
            print(f"Sample: {list(only_in_ground_truth)[:5]}")

        # Show overlap percentage
        if prediction_event_ids:
            overlap_pct = (len(matching_ids) / len(prediction_event_ids)) * 100
            print(f"\n📈 Overlap: {overlap_pct:.1f}% of prediction event IDs found in ground truth")

    # Show sample rows from both datasets
    print("\n" + "=" * 80)
    print("SAMPLE DATA")
    print("=" * 80)

    print("\nPredictions (first 5 rows):")
    print(predictions_df.head())

    print("\nGround Truth (first 5 rows):")
    print(ground_truth_df.head())

except Exception as e:
    print(f"\n❌ Error accessing ground truth: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("COMPLETE")
print("=" * 80)
