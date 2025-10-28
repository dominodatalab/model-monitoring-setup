#!/usr/bin/env python3
"""
Setup Model Monitoring Configuration

Interactive setup to configure model monitoring.

Usage:
    python setup_monitoring.py
"""

import json
import sys
from pathlib import Path


def get_input(prompt: str, default: str = None, required: bool = True) -> str:
    """Get user input with optional default"""
    if default:
        full_prompt = f"{prompt} [{default}]: "
    else:
        full_prompt = f"{prompt}: "

    while True:
        value = input(full_prompt).strip()

        if value:
            return value
        elif default:
            return default
        elif not required:
            return ""
        else:
            print("Required field. Please enter a value.")


def create_config():
    """Interactive configuration creation"""
    print("🔧 MODEL MONITORING SETUP")
    print("=" * 50)

    while True:
        # Domino Configuration
        print("\n📍 DOMINO CONFIGURATION")
        print("-" * 30)
        domino_url = get_input(
            "Domino base URL (e.g., https://your-domino.com)",
            "https://your-domino-instance.com"
        ).rstrip('/')

        # Model API Configuration
        print("\n🔗 MODEL API CONFIGURATION")
        print("-" * 30)
        model_api_url = get_input(
            "Model API endpoint URL",
            f"{domino_url}:443/models/YOUR_MODEL_ID/labels/prod/model"
        )

        model_api_token = get_input(
            "Model API token",
            required=False
        )
        if not model_api_token:
            model_api_token = "REPLACE_WITH_YOUR_TOKEN"

        # Model Monitor Configuration
        print("\n📊 MODEL MONITOR CONFIGURATION")
        print("-" * 30)
        model_monitor_id = get_input(
            "Model Monitor ID",
            "YOUR_MODEL_MONITOR_ID"
        )

        # Data Sources
        print("\n💾 DATA SOURCE CONFIGURATION")
        print("-" * 30)
        ground_truth_ds = get_input(
            "Ground truth data source name",
            "your-ground-truth-datasource"
        )

        # Test Data
        print("\n📁 TEST DATA CONFIGURATION")
        print("-" * 30)
        test_data_path = get_input(
            "Path to test data directory (for generating predictions)",
            "/mnt/data/test",
            required=False
        )

        # Create config
        config = {
            "_comment": "Model Monitoring Configuration",
            "domino": {
                "base_url": domino_url,
                "user_api_key_env": "DOMINO_USER_API_KEY"
            },
            "model_api": {
                "endpoint_url": model_api_url,
                "token": model_api_token,
                "timeout": 60
            },
            "model_monitor": {
                "model_id": model_monitor_id
            },
            "data_sources": {
                "ground_truth": ground_truth_ds
            },
            "test_data": {
                "path": test_data_path
            },
            "monitoring": {
                "daily_predictions": 30
            }
        }

        # Display for verification
        print("\n" + "=" * 50)
        print("📋 CONFIGURATION SUMMARY")
        print("=" * 50)
        print(f"🌐 Domino URL: {domino_url}")
        print(f"🔗 Model API: {model_api_url}")

        if model_api_token != "REPLACE_WITH_YOUR_TOKEN":
            masked = f"****...{model_api_token[-4:]}" if len(model_api_token) > 4 else "****"
            print(f"🔑 API Token: {masked}")
        else:
            print(f"🔑 API Token: [NOT SET]")

        print(f"📊 Monitor ID: {model_monitor_id}")
        print(f"💾 Ground Truth DS: {ground_truth_ds}")
        print(f"📁 Test Data: {test_data_path or '[Not set]'}")
        print("=" * 50)

        confirmation = input("\n✅ Is this correct? (yes/no/cancel): ").strip().lower()

        if confirmation in ['yes', 'y']:
            return config
        elif confirmation in ['cancel', 'c', 'quit', 'q']:
            print("\n❌ Setup cancelled")
            return None
        else:
            print("\n🔄 Starting over...\n")
            continue


def save_config(config):
    """Save configuration to file"""
    config_file = Path('monitoring_config.json')

    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✅ Config saved: {config_file}")
    return config_file


def main():
    try:
        config = create_config()

        if config is None:
            sys.exit(1)

        save_config(config)

        print("\n🎉 SETUP COMPLETE!")
        print("=" * 50)

        step = 1
        if config['model_api']['token'] == "REPLACE_WITH_YOUR_TOKEN":
            print(f"{step}. Update API token in monitoring_config.json")
            step += 1

        print(f"{step}. Run monitoring scripts:")
        print("   python upload_ground_truth.py")
        print("   python generate_predictions.py")

        print("\n" + "="*50)
        print("✅ MONITORING CONFIGURATION COMPLETE")
        print("="*50)
        print("\nNext step:")
        print("python 4_generate_predictions.py --count 30")
        print("="*50)

    except KeyboardInterrupt:
        print("\n\n❌ Cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
