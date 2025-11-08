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
        
        # Track retry attempts to prevent infinite loops
        self.retry_count = 0
        self.max_retries = 3

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

    def get_file_columns(self, file_path):
        """Get available columns from the data file for validation"""
        try:
            import pandas as pd
            
            # Check if this is a remote path (starts with data source path patterns)
            if file_path.startswith('ground_truth/') or file_path.startswith('s3://') or '/' in file_path:
                # For remote files, return columns that are actually created by 4_generate_predictions.py
                print(f"   📡 Remote file detected, using actual ground truth file structure")
                return ["event_id", "target", "timestamp"]
            
            # Try to read local file
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, nrows=1)
            elif file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
                df = df.head(1)
            else:
                return ["event_id", "target", "timestamp"]  # Default fallback matching actual data
                
            return list(df.columns)
        except Exception as e:
            print(f"   ⚠️  Could not read file columns from {file_path}: {e}")
            # Return actual ground truth columns as fallback
            return ["event_id", "target", "timestamp"]

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
        
        # Get available columns from the file
        available_columns = self.get_file_columns(file_path)
        print(f"   📋 Available columns in data: {available_columns}")
        
        # Choose ground truth column name based on what's available
        gt_column = self.ground_truth_column
        if available_columns:
            # Use actual columns from ground truth data
            if "target" in available_columns:
                gt_column = "target"
            elif self.ground_truth_column in available_columns:
                gt_column = self.ground_truth_column
            else:
                gt_column = self.ground_truth_column  # fallback
            print(f"   🎯 Using ground truth column: {gt_column}")
        
        # Choose row identifier column
        row_id_column = "event_id"
        if available_columns:
            if "event_id" in available_columns:
                row_id_column = "event_id"
            else:
                # Keep event_id as default since it's what our prediction capture uses
                row_id_column = "event_id"
            print(f"   🔑 Using row identifier column: {row_id_column}")

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
                    "name": row_id_column,
                    "variableType": "row_identifier",
                    "valueType": "string"
                },
                {
                    "name": gt_column,
                    "variableType": "ground_truth",
                    "valueType": self.value_type
                }
            ]
        }

    def create_unique_config(self, original_config, file_path, removed_variables):
        """Create a new config with unique name to avoid conflicts"""
        import uuid
        import copy
        
        # Create deep copy of original config
        new_config = copy.deepcopy(original_config)
        
        # Generate unique suffix
        unique_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        
        # Get original name and create new unique name
        original_name = original_config["datasetDetails"]["name"]
        
        # Add retry info to the name
        retry_info = ""
        if removed_variables:
            retry_info = f"_retry_no_{'-'.join(removed_variables).replace(' ', '')}"
        else:
            retry_info = "_retry_no_vars"
        
        new_name = f"{original_name}{retry_info}_{unique_id}"
        
        # Update the dataset name
        new_config["datasetDetails"]["name"] = new_name
        
        print(f"   📝 Created unique config name: {new_name}")
        return new_config

    def retry_with_variable_removal(self, config, file_path, error_message):
        """Retry upload by removing variables mentioned in error message"""
        import re
        
        # Check retry limit
        self.retry_count += 1
        if self.retry_count > self.max_retries:
            print(f"   ❌ Maximum retries ({self.max_retries}) exceeded")
            print(f"   🔄 Final attempt: trying with no variables (datasetDetails only)")
            
            # Final fallback: try with no variables
            new_config = self.create_unique_config(config, file_path, ["final_attempt"])
            new_config.pop("variables", None)  # Remove all variables
            
            print(f"   📤 Final upload: {new_config['datasetDetails']['name']}")
            print(f"   📊 Variables: 0 configured (datasetDetails only)")
            
            # Print JSON configuration for final attempt debugging
            import json
            print(f"   📄 Final Attempt Configuration JSON:")
            print(json.dumps(new_config, indent=2))
            
            try:
                url = f"{self.base_url}/model/{self.config.model_monitor_id}/register-dataset/ground_truth"
                response = requests.put(url, headers=self.headers, json=new_config, timeout=60)
                
                if response.status_code in [200, 201]:
                    print(f"✅ Registered with no variables: {file_path}")
                    return True
                else:
                    print(f"❌ Final attempt failed: {response.status_code} - {response.text}")
                    return False
            except Exception as e:
                print(f"❌ Final attempt error: {e}")
                return False
        
        print(f"   🔍 Retry {self.retry_count}/{self.max_retries} - Analyzing error: {error_message}")
        
        # Extract variable names from error message - handle multiple patterns
        conflicts = []
        
        # Pattern 1: "Variable(s) with name ['target', 'event_id'] is/are already configured"
        pattern1 = r"Variable\(s\) with name \[([^\]]+)\]"
        match1 = re.search(pattern1, error_message)
        if match1:
            # Extract individual variable names from the list
            var_list = match1.group(1)
            # Handle both quoted and unquoted variable names
            var_names = re.findall(r"'([^']+)'", var_list)
            if not var_names:
                # Try without quotes
                var_names = [name.strip() for name in var_list.split(',')]
            conflicts.extend(var_names)
        
        # Pattern 2: Single variable mentions
        pattern2 = r"Variable '([^']+)' (?:is already|already)"
        conflicts.extend(re.findall(pattern2, error_message, re.IGNORECASE))
        
        if conflicts:
            print(f"   🔄 Found conflicting variables: {conflicts}")
            
            # Check which types of variables we have
            original_vars = config.get("variables", [])
            ground_truth_vars = [v for v in original_vars if v.get("variableType") == "ground_truth"]
            row_id_vars = [v for v in original_vars if v.get("variableType") == "row_identifier"]
            
            # Strategy 1: Try to use different column names for conflicting variables
            new_config = self.create_unique_config(config, file_path, conflicts)
            retry_vars = []
            
            # Get available columns from the data file
            available_columns = self.get_file_columns(file_path)
            print(f"   📋 Available columns for retry: {available_columns}")
            
            # Handle conflicts by removing conflicting variables, not renaming them
            print(f"   🔄 Removing conflicting variables: {conflicts}")
            
            # Keep only non-conflicting variables
            non_conflicting_gt_vars = [v for v in ground_truth_vars if v["name"] not in conflicts]
            non_conflicting_row_id_vars = [v for v in row_id_vars if v["name"] not in conflicts]
            
            # Add non-conflicting variables to retry config
            retry_vars.extend(non_conflicting_gt_vars)
            retry_vars.extend(non_conflicting_row_id_vars)
            
            # If we have variables left, try with them
            if retry_vars:
                new_config["variables"] = retry_vars
                print(f"   🔄 Retrying with {len(retry_vars)} non-conflicting variables")
                for var in retry_vars:
                    print(f"      - {var['name']} ({var['variableType']})")
                return self.upload_config_simple(new_config, file_path)
            else:
                # If all variables conflict, try with no variables (dataset only)
                print(f"   💡 All variables conflict - trying with dataset registration only (no variable mappings)")
                new_config["variables"] = []
                return self.upload_config_simple(new_config, file_path)
        else:
            print("   ❌ Could not parse variable conflicts from error message")
            return False

    def retry_without_row_identifier(self, config, file_path):
        """Retry upload with only ground truth variable (no row identifier)"""
        # Check retry limit
        self.retry_count += 1
        if self.retry_count > self.max_retries:
            print(f"   ❌ Maximum retries ({self.max_retries}) exceeded - stopping")
            return False
            
        print(f"   🔄 Retry {self.retry_count}/{self.max_retries} - Retrying with ground truth variable only (no row identifier)")
        
        # Create new config with unique name and only ground truth variable
        new_config = self.create_unique_config(config, file_path, ["row_identifier"])
        
        # Keep only ground truth variables
        original_vars = config.get("variables", [])
        ground_truth_vars = [v for v in original_vars if v.get("variableType") == "ground_truth"]
        
        if ground_truth_vars:
            new_config["variables"] = ground_truth_vars
            print(f"   🎯 Using only ground truth variable: {ground_truth_vars[0]['name']}")
        else:
            # Create a basic ground truth variable
            available_columns = self.get_file_columns(file_path)
            # Use 'target' as primary choice since that's what 4_generate_predictions.py creates
            if "target" in available_columns:
                gt_column = "target"
            elif "actual_class" in available_columns:
                gt_column = "actual_class"  
            else:
                gt_column = self.ground_truth_column
            new_config["variables"] = [{
                "name": gt_column,
                "variableType": "ground_truth",
                "valueType": self.value_type
            }]
            print(f"   🎯 Created ground truth variable: {gt_column}")
        
        return self.upload_config_simple(new_config, file_path)

    def upload_config_simple(self, config, file_path):
        """Upload configuration without complex error handling (for retries)"""
        url = f"{self.base_url}/model/{self.config.model_monitor_id}/register-dataset/ground_truth"
        
        print(f"   📤 Retry upload: {config['datasetDetails']['name']}")
        print(f"   📊 Variables: {len(config.get('variables', []))} configured")
        for var in config.get('variables', []):
            print(f"      - {var['name']} ({var['variableType']})")
        
        # Print JSON configuration for debugging retries
        import json
        print(f"   📄 Retry Configuration JSON:")
        print(json.dumps(config, indent=2))
        
        try:
            response = requests.put(url, headers=self.headers, json=config, timeout=60)
            
            if response.status_code in [200, 201]:
                print(f"✅ Registered: {file_path}")
                return True
            elif response.status_code == 409:
                print(f"ℹ️  Already registered: {file_path}")
                return True
            elif response.status_code == 400:
                if "can be only 1 row_identifier" in response.text.lower():
                    print(f"   ℹ️  Row identifier conflict in retry - trying without row identifier")
                    return self.retry_without_row_identifier(config, file_path)
                elif "already configured" in response.text.lower():
                    print(f"   ℹ️  Variables still conflicting in retry - trying with different names")
                    return self.retry_with_variable_removal(config, file_path, response.text)
                else:
                    print(f"❌ Retry failed: {response.status_code} - {response.text}")
                    return False
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
        print(f"   📋 Dataset name: {config['datasetDetails']['name']}")
        print(f"   📊 Variables: {len(config.get('variables', []))} configured")
        for var in config.get('variables', []):
            print(f"      - {var['name']} ({var['variableType']})")
        
        # Print JSON configuration for debugging
        import json
        print(f"   📄 Configuration JSON:")
        print(json.dumps(config, indent=2))

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
                        print("      Check that 'target' column exists in the ground truth CSV")
                        print("      and that variable types are correctly defined")
                    elif "can be only 1 row_identifier" in response.text.lower():
                        print("   ℹ️  Row identifier already exists globally - retrying without row identifier")
                        return self.retry_without_row_identifier(config, file_path)
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
            # Reset retry count for each file
            self.retry_count = 0
            
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
