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

        files = list(test_path.rglob("*.*"))
        files = [f for f in files if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.csv']]

        print(f"📂 Found {len(files)} test files")
        return files

    def call_model_api(self, file_path):
        """
        Call deployed Model API.

        Customize this method based on your model's input format.
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
                payload = {'data': df.to_dict('records')[0]}

            # Send request with HTTP Basic Auth
            response = requests.post(
                self.model_api_url,
                json=payload,
                auth=(self.model_api_token, self.model_api_token),
                timeout=60
            )

            response.raise_for_status()
            result = response.json().get('result', response.json())

            if 'label' not in result or 'score' not in result:
                return {'error': f'Invalid response: {response.json()}'}

            return result

        except Exception as e:
            return {'error': f'{e}'}

    def generate_predictions(self, count=30):
        """Generate predictions from test data"""
        print("=" * 60)
        print("PREDICTION GENERATION")
        print("=" * 60)
        print(f"Target: {count} predictions")

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

                # Determine actual class from filename/folder structure
                # Customize this based on your data organization
                actual_class = file_path.parent.name

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
                    'actual_class': actual_class,
                    'predicted_class': result.get('label'),
                    'confidence': result.get('score')
                })

                successful += 1

                if (i + 1) % 10 == 0:
                    print(f"   Progress: {i + 1}/{count}")

                time.sleep(random.uniform(1, 3))

            except Exception as e:
                print(f"   ❌ Error: {e}")
                failed += 1

        print(f"\n✅ Complete:")
        print(f"   Successful: {successful}")
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
                    'actual_class': pred['actual_class'],
                    'timestamp': pred['timestamp']
                })

            csv_buffer = io.StringIO()
            writer = csv.DictWriter(csv_buffer, fieldnames=['event_id', 'actual_class', 'timestamp'])
            writer.writeheader()
            writer.writerows(records)

            csv_bytes = csv_buffer.getvalue().encode('utf-8')

            # Upload
            print(f"📤 Uploading to: {self.datasource_name}")
            print(f"   Records: {len(records)}")

            datasource = DataSourceClient().get_datasource(self.datasource_name)
            s3_key = f"ground_truth/{self.model_id}/{self.today}.csv"

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
            print(f"   python upload_ground_truth.py")

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
