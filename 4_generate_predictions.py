#!/usr/bin/env python3
"""
Generate Predictions and Ground Truth Data

Call the deployed model API with test data to generate predictions and
upload corresponding ground truth for Model Monitor.

Usage:
    python generate_predictions.py
    python generate_predictions.py --count 50
"""

import os
import sys
import argparse
import random
import time
import csv
import io
import base64
import requests
from datetime import datetime, timezone
from pathlib import Path

from config_loader import get_config

try:
    from domino.data_sources import DataSourceClient
    DATASOURCE_AVAILABLE = True
except ImportError:
    DATASOURCE_AVAILABLE = False
    print("⚠️  domino.data_sources not available")


class PredictionGenerator:
    """Generate predictions and ground truth"""

    def __init__(self):
        self.config = get_config()
        self.today = datetime.now().strftime('%Y-%m-%d')
        self.predictions = []

        self.model_api_url = self.config.model_api_url
        self.model_api_token = self.config.model_api_token
        self.datasource_name = self.config.ground_truth_datasource
        self.model_id = self.config.model_monitor_id

    def get_test_files(self):
        """Load test files from configured path"""
        test_path = Path(self.config.test_data_path)

        if not test_path.exists():
            print(f"❌ Test data path not found: {test_path}")
            return []

        files = []
        if test_path.is_file():
            # Single file specified
            if test_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.csv']:
                files = [test_path]
        elif test_path.is_dir():
            # Directory specified - search for files
            files = list(test_path.rglob("*.*"))
            files = [f for f in files if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.csv']]

        print(f"📂 Found {len(files)} test files")
        return files

    def call_model_api(self, file_path):
        """
        Call deployed Model API.

        CUSTOMIZE THIS METHOD based on your model's input format:
        1. Update payload structure to match your model's expected input
        2. Modify response parsing to extract prediction results
        3. Adjust authentication if using different token format
        """
        try:
            # Example for image input (base64 encoded)
            if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                with open(file_path, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')

                payload = {'data': {'image': image_data}}

            # Example for CSV/tabular input
            else:
                # Read first row of CSV as example
                import pandas as pd
                df = pd.read_csv(file_path, nrows=1)
                # CUSTOMIZE: Modify this to match your model's input format
                payload = {'data': df.to_dict('records')[0]}

            # Send request with HTTP Basic Auth
            # CUSTOMIZE: Update authentication method if needed
            response = requests.post(
                self.model_api_url,
                json=payload,
                auth=(self.model_api_token, self.model_api_token),
                timeout=60
            )

            response.raise_for_status()
            json_response = response.json()
            result = json_response.get('result', json_response)

            # CUSTOMIZE: Update response parsing to match your model's output format
            # Handle different response formats
            if 'predicted_class' in result and 'confidence_score' in result:
                # Example model format
                return result
            elif 'label' in result and 'score' in result:
                # Generic format - map to expected keys
                return {
                    'predicted_class': result['label'],
                    'confidence_score': result['score'],
                    'event_id': result.get('event_id'),
                    'timestamp': result.get('timestamp')
                }
            else:
                return {'error': f'Invalid response format: {json_response}'}

        except Exception as e:
            return {'error': f'{e}'}

    def generate_predictions(self, count=30):
        """Generate predictions from test data"""
        print("=" * 60)
        print("PREDICTION GENERATION")
        print("=" * 60)
        print(f"Target: {count} predictions")
        print(f"API URL: {self.model_api_url}")

        test_files = self.get_test_files()
        if not test_files:
            print("❌ No test files found")
            return False

        successful = 0
        failed = 0

        for i in range(count):
            try:
                # Randomly select a test file
                file_path = random.choice(test_files)

                # CUSTOMIZE: Determine actual ground truth based on your data organization
                # Option 1: Extract from folder structure (current approach)
                actual_class = file_path.parent.name
                
                # Option 2: Extract from CSV data (if using tabular data)
                # if file_path.suffix.lower() == '.csv':
                #     import pandas as pd
                #     df = pd.read_csv(file_path, nrows=1)
                #     actual_class = df['target'].iloc[0]  # Adjust column name as needed
                
                # Option 3: Use filename pattern
                # actual_class = file_path.stem.split('_')[0]  # Extract prefix before underscore
                
                # Option 4: Use lookup table/mapping
                # actual_class = self.get_ground_truth_for_file(file_path)

                # Call API
                result = self.call_model_api(file_path)

                if 'error' in result:
                    print(f"   ❌ API error: {result['error']}")
                    failed += 1
                    continue

                # Store prediction
                self.predictions.append({
                    'event_id': result.get('event_id', f'pred_{i}'),
                    'timestamp': result.get('timestamp', datetime.now(timezone.utc).isoformat()),
                    'target': actual_class,  # CUSTOMIZE: Column name must match your training data
                    'predicted_class': result.get('predicted_class'),
                    'confidence': result.get('confidence_score')
                })

                successful += 1

                if (i + 1) % 10 == 0:
                    print(f"   Progress: {i + 1}/{count}")

                time.sleep(random.uniform(1, 3))

            except Exception as e:
                print(f"   ❌ Error: {e}")
                failed += 1

        print(f"\n✅ Complete: {successful} successful API calls")
        print(f"   Failed: {failed}")

        return successful > 0

    def upload_ground_truth(self):
        """Upload ground truth to data source"""
        if not self.predictions:
            print("⚠️  No predictions to upload")
            return False

        if not DATASOURCE_AVAILABLE:
            print("⚠️  Cannot upload - DataSourceClient unavailable")
            return False

        try:
            print("\n" + "=" * 60)
            print("GROUND TRUTH UPLOAD")
            print("=" * 60)

            # Create CSV
            records = []
            for pred in self.predictions:
                records.append({
                    'event_id': pred['event_id'],
                    'target': pred['target'],  # Changed from 'actual_class' to 'target'
                    'timestamp': pred['timestamp']
                })

            csv_buffer = io.StringIO()
            writer = csv.DictWriter(csv_buffer, fieldnames=['event_id', 'target', 'timestamp'])
            writer.writeheader()
            writer.writerows(records)

            csv_bytes = csv_buffer.getvalue().encode('utf-8')

            # Upload
            print(f"📤 Uploading to: {self.datasource_name}")
            print(f"   Records: {len(records)}")

            datasource = DataSourceClient().get_datasource(self.datasource_name)
            s3_key = f"ground_truth/{self.model_id}/{self.today}.csv"
            
            print(f"Ground truth file location: {self.datasource_name}")
            print(f"Ground truth file name: {s3_key}")

            bytes_buffer = io.BytesIO(csv_bytes)
            datasource.upload_fileobj(s3_key, bytes_buffer)

            print(f"✅ Uploaded: {s3_key}")
            return True

        except Exception as e:
            print(f"❌ Upload failed: {e}")
            return False

    def run(self, count=30):
        """Main execution"""
        success = self.generate_predictions(count)

        if not success:
            return 1

        gt_uploaded = self.upload_ground_truth()

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Predictions: {len(self.predictions)}")
        print(f"Ground truth uploaded: {'Yes' if gt_uploaded else 'No'}")

        if gt_uploaded:
            print(f"\n⏰ Wait 1 hour, then run:")
            print(f"   python 5_upload_ground_truth.py")

        return 0


def main():
    parser = argparse.ArgumentParser(description="Generate predictions and ground truth")
    parser.add_argument("--count", type=int, default=30, help="Number of predictions")

    args = parser.parse_args()

    try:
        generator = PredictionGenerator()
        sys.exit(generator.run(args.count))

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    print("\n" + "="*50)
    print("✅ PREDICTIONS GENERATED SUCCESSFULLY")
    print("="*50)
    print("\nNext step (WAIT 1+ HOURS FIRST):")
    print("python 5_upload_ground_truth.py")
    print("\nNote: Ground truth data needs time to upload to S3")
    print("="*50)


if __name__ == "__main__":
    main()
