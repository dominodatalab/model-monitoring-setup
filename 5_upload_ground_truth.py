#!/usr/bin/env python3
"""
Upload Ground Truth Configuration to Model Monitor

Register ground truth datasets with Model Monitor for quality metrics.

Usage:
    python upload_ground_truth.py
    python upload_ground_truth.py --hours 168  # Last 7 days
    python upload_ground_truth.py --start-date 2025-10-01 --end-date 2025-10-27

IMPORTANT: Run this at least 1 hour after predictions are generated to allow
           ground truth data to be uploaded to the data source.
"""

import os
import sys
import json
import argparse
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path

from config_loader import get_config


class GroundTruthUploader:
    """
    Upload ground truth configurations to Model Monitor
    
    CUSTOMIZE: Update the class for your specific use case:
    - Modify ground_truth_column_name if different from 'target'
    - Change is_regression=True for numerical targets
    """

    def __init__(self, ground_truth_column_name="target", is_regression=False, datasource_type="s3", force_reregister=False):
        self.config = get_config()
        self.base_url = self.config.model_monitor_api_url
        self.headers = {
            "Content-Type": "application/json",
            "X-Domino-Api-Key": self.config.domino_api_key
        }
        
        # CUSTOMIZE: Set these based on your model type
        self.ground_truth_column = ground_truth_column_name
        self.value_type = "numerical" if is_regression else "categorical"
        self.datasource_type = datasource_type
        self.force_reregister = force_reregister

        print(f"🔗 Connecting to: {self.config.domino_base_url}")
        print(f"📊 Model Monitor ID: {self.config.model_monitor_id}")
        print(f"💾 Data Source: {self.config.ground_truth_datasource}")
        print(f"🎯 Ground Truth Column: {self.ground_truth_column} ({self.value_type})")
        if force_reregister:
            print("🔄 Force re-register mode enabled")

    def discover_files(self, start_date=None, end_date=None, hours_back=24):
        """Discover ground truth files in date range"""
        if not start_date or not end_date:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(hours=hours_back)

        print(f"🔍 Searching {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        file_paths = []
        current = start_date.date()
        end = end_date.date()

        while current <= end:
            path = f"ground_truth/{self.config.model_monitor_id}/{current.strftime('%Y-%m-%d')}.csv"
            file_paths.append(path)
            current += timedelta(days=1)

        print(f"📂 Generated {len(file_paths)} file paths")
        return file_paths

    def check_existing_dataset(self, dataset_name):
        """Check if dataset already exists"""
        try:
            url = f"{self.base_url}/models/{self.config.model_monitor_id}/groundtruth-datasets"
            response = requests.get(url, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                datasets = response.json()
                for dataset in datasets:
                    if dataset.get('name') == dataset_name:
                        return dataset
            return None
        except Exception as e:
            print(f"⚠️  Could not check existing datasets: {e}")
            return None

    def delete_existing_dataset(self, dataset_id):
        """Delete an existing dataset"""
        try:
            url = f"{self.base_url}/models/{self.config.model_monitor_id}/groundtruth-datasets/{dataset_id}"
            response = requests.delete(url, headers=self.headers, timeout=30)
            
            if response.status_code in [200, 204]:
                print(f"   ✅ Deleted existing dataset: {dataset_id}")
                return True
            else:
                print(f"   ⚠️  Failed to delete dataset {dataset_id}: {response.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ Error deleting dataset {dataset_id}: {e}")
            return False

    def create_config(self, file_path):
        """
        Create ground truth configuration payload
        
        CUSTOMIZE: Update this method based on your ground truth data structure:
        1. Modify ground truth column name to match your data
        2. Change valueType if using numerical/regression targets
        """
        filename = Path(file_path).name
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        dataset_name = f"ground_truth_{self.config.model_monitor_id}_{filename.replace('.csv', '')}_{timestamp}"

        return {
            "datasetDetails": {
                "name": dataset_name,
                "datasetType": "file",
                "datasetConfig": {
                    "path": file_path,
                    "fileFormat": "csv"  # CUSTOMIZE: Change if using different format
                },
                "datasourceName": self.config.ground_truth_datasource,
                "datasourceType": self.datasource_type
            },
            "variables": [
                {
                    "name": "event_id",
                    "variableType": "row_identifier",
                    "valueType": "string"
                },
                {
                    "name": self.ground_truth_column,
                    "variableType": "ground_truth",
                    "valueType": self.value_type
                }
            ]
        }

    def retry_with_variable_removal(self, config, file_path, error_message):
        """Retry upload by removing variables mentioned in error message"""
        import re
        
        # Extract variable names from error message
        variable_pattern = r"Variable\(s\) with name \['([^']+)'\]"
        matches = re.findall(variable_pattern, error_message)
        
        if matches:
            print(f"   🔄 Removing already configured variables: {matches}")
            # Remove the mentioned variables
            remaining_vars = [var for var in config.get("variables", []) 
                            if var["name"] not in matches]
            
            if remaining_vars:
                # Try with remaining variables
                new_config = config.copy()
                new_config["variables"] = remaining_vars
                print(f"   🔄 Retrying with {len(remaining_vars)} variables")
                return self.upload_config_simple(new_config, file_path)
            else:
                # Try without any variables
                print("   🔄 Retrying without variable definitions")
                config_no_vars = {"datasetDetails": config["datasetDetails"]}
                return self.upload_config_simple(config_no_vars, file_path)
        else:
            # Try without any variables as fallback
            print("   🔄 Retrying without variable definitions")
            config_no_vars = {"datasetDetails": config["datasetDetails"]}
            return self.upload_config_simple(config_no_vars, file_path)

    def upload_config_simple(self, config, file_path):
        """Upload configuration without complex error handling (for retries)"""
        url = f"{self.base_url}/model/{self.config.model_monitor_id}/register-dataset/ground_truth"
        
        try:
            response = requests.put(url, headers=self.headers, json=config, timeout=60)
            
            if response.status_code in [200, 201]:
                print(f"✅ Registered: {file_path}")
                return True
            elif response.status_code == 409:
                print(f"ℹ️  Already registered: {file_path}")
                return True
            else:
                print(f"❌ Retry failed: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error on retry: {e}")
            return False

    def upload_config(self, config):
        """Upload configuration to Model Monitor API"""
        url = f"{self.base_url}/model/{self.config.model_monitor_id}/register-dataset/ground_truth"
        file_path = config["datasetDetails"]["datasetConfig"]["path"]

        print(f"📤 Uploading: {file_path}")

        try:
            response = requests.put(url, headers=self.headers, json=config, timeout=60)

            if response.status_code in [200, 201]:
                print(f"✅ Registered: {file_path}")
                return True
            elif response.status_code == 409:
                print(f"ℹ️  Already registered: {file_path}")
                return True
            else:
                print(f"❌ Failed: {file_path}")
                print(f"   Status: {response.status_code}")
                print(f"   Response: {response.text}")
                
                # Provide specific guidance for common errors
                if response.status_code == 400:
                    if "already configured" in response.text.lower():
                        print("   ℹ️  Some variables are already configured - retrying with adjustments")
                        return self.retry_with_variable_removal(config, file_path, response.text)
                    elif "ground truth variable" in response.text.lower():
                        print("   💡 Issue: Ground truth variable configuration problem")
                        print("      Check that 'actual_class' column exists in the ground truth CSV")
                        print("      and that variable types are correctly defined")
                    elif "row_identifier" in response.text.lower():
                        print("   💡 Issue: Row identifier configuration problem")
                        print("      Check that 'event_id' column exists and uniquely identifies rows")
                elif response.status_code == 404:
                    print("   💡 Issue: Model Monitor ID or data source not found")
                    print(f"      Verify Model Monitor ID: {self.config.model_monitor_id}")
                    print(f"      Verify data source: {self.config.ground_truth_datasource}")
                elif response.status_code == 403:
                    print("   💡 Issue: Permission denied")
                    print("      Check API key permissions for Model Monitor")
                
                return False

        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            return False

    def upload_range(self, start_date=None, end_date=None, hours_back=24):
        """Upload all files in date range"""
        file_paths = self.discover_files(start_date, end_date, hours_back)

        if not file_paths:
            print("⚠️  No files found")
            return {"successful": 0, "failed": 0, "total": 0}

        successful = 0
        failed = 0

        print(f"\n🚀 Starting upload ({len(file_paths)} files)...")

        for file_path in file_paths:
            config = self.create_config(file_path)
            if self.upload_config(config):
                successful += 1
            else:
                failed += 1

        print(f"\n📊 Summary:")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")

        return {"successful": successful, "failed": failed, "total": len(file_paths)}


def parse_date(date_str):
    """Parse YYYY-MM-DD format"""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date: {date_str}. Use YYYY-MM-DD")


def main():
    parser = argparse.ArgumentParser(description="Upload ground truth to Model Monitor")
    parser.add_argument("--hours", type=int, default=24, help="Hours back (default: 24)")
    parser.add_argument("--start-date", type=parse_date, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=parse_date, help="End date (YYYY-MM-DD)")
    
    # CUSTOMIZE: Add these arguments for your specific model
    parser.add_argument("--ground-truth-column", default="target", 
                       help="Ground truth column name (default: target)")
    parser.add_argument("--regression", action="store_true", 
                       help="Use for regression models (numerical targets)")
    parser.add_argument("--datasource-type", default="s3", 
                       help="Data source type (default: s3)")
    parser.add_argument("--force-reregister", action="store_true", 
                       help="Force re-registration of existing datasets")

    args = parser.parse_args()

    if (args.start_date or args.end_date) and not (args.start_date and args.end_date):
        parser.error("Both --start-date and --end-date required")

    print("🔬 Ground Truth Upload")
    print("=" * 60)

    try:
        # CUSTOMIZE: Create uploader with your model-specific settings
        uploader = GroundTruthUploader(
            ground_truth_column_name=args.ground_truth_column,
            is_regression=args.regression,
            datasource_type=args.datasource_type,
            force_reregister=args.force_reregister
        )

        if args.start_date and args.end_date:
            results = uploader.upload_range(args.start_date, args.end_date)
        else:
            results = uploader.upload_range(hours_back=args.hours)

        if results["failed"] > 0:
            sys.exit(1)
        elif results["successful"] > 0:
            print(f"\n🎉 All {results['successful']} configurations uploaded!")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    print("\n" + "="*50)
    print("✅ GROUND TRUTH UPLOAD COMPLETE")
    print("="*50)
    print("\nSetup complete! Check Model Monitor UI:")
    print("- Predictions: Model Details > Predictions")
    print("- Ground Truth: Ground Truth Status")
    print("- Schedule your automated checks in the Model Monitor to see Metrics")
    print("="*50)


if __name__ == "__main__":
    main()
