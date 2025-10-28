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
    """Upload ground truth configurations to Model Monitor"""

    def __init__(self):
        self.config = get_config()
        self.base_url = self.config.model_monitor_api_url
        self.headers = {
            "Content-Type": "application/json",
            "X-Domino-Api-Key": self.config.domino_api_key
        }

        print(f"🔗 Connecting to: {self.config.domino_base_url}")
        print(f"📊 Model Monitor ID: {self.config.model_monitor_id}")
        print(f"💾 Data Source: {self.config.ground_truth_datasource}")

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

    def create_config(self, file_path):
        """Create ground truth configuration payload"""
        filename = Path(file_path).name
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        dataset_name = f"ground_truth_{self.config.model_monitor_id}_{filename.replace('.csv', '')}_{timestamp}"

        return {
            "datasetDetails": {
                "name": dataset_name,
                "datasetType": "file",
                "datasetConfig": {
                    "path": file_path,
                    "fileFormat": "csv"
                },
                "datasourceName": self.config.ground_truth_datasource,
                "datasourceType": "s3"
            }
        }

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

    args = parser.parse_args()

    if (args.start_date or args.end_date) and not (args.start_date and args.end_date):
        parser.error("Both --start-date and --end-date required")

    print("🔬 Ground Truth Upload")
    print("=" * 60)

    try:
        uploader = GroundTruthUploader()

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
