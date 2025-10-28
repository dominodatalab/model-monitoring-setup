# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Domino Model Monitor setup toolkit** - a collection of Python scripts for configuring automated prediction capture and ground truth tracking in Domino Data Lab environments.

## Key Scripts and Workflow

The project follows a sequential workflow for setting up model monitoring:

1. **`1_register_training_set.py`** - Register training data as baseline for drift detection
2. **`prediction_capture_example.py`** - Template for adding monitoring to model deployment scripts
3. **`2_export_model.py`** - Export models from MLflow/Domino Model Registry (optional)
4. **`3_setup_monitoring.py`** - Interactive configuration wizard
5. **`4_generate_predictions.py`** - Generate test predictions and upload ground truth
6. **`5_upload_ground_truth.py`** - Register ground truth datasets (run 1+ hours after predictions)

## Core Architecture

### Configuration Management
- **`config_loader.py`** - Centralized configuration through `MonitoringConfig` class
- **`monitoring_config.json.template`** - Configuration template
- Configuration automatically searches multiple paths: `./`, `/mnt/`, `/mnt/artifacts/`
- Uses `DOMINO_USER_API_KEY` environment variable (auto-available in Domino workspaces)

### Key Dependencies
- `domino_data_capture.data_capture_client` - For prediction capture
- `domino.training_sets` - For training set registration  
- Standard Python libraries: `requests`, `pandas`, `mlflow`, `joblib`

## Running Scripts

### Basic Commands
```bash
# Register training set (required unique name across Domino instance)
python 1_register_training_set.py --file /path/to/training.csv --name "My Training Set"

# Interactive setup
python 3_setup_monitoring.py

# Generate test predictions
python 4_generate_predictions.py --count 30

# Upload ground truth (wait 1+ hours after predictions)
python 5_upload_ground_truth.py --hours 168
```

### Model Export
```bash
# Export by metric
python 2_export_model.py --metric accuracy

# Export by run ID
python 2_export_model.py --run-id YOUR_RUN_ID
```

## Environment Context

This project is designed specifically for **Domino Data Lab** environments:
- Automatic API key access via `DOMINO_USER_API_KEY`
- Integration with Domino Model APIs and Model Monitor
- Data source configuration across Domino UI, Project Settings, and Model Monitor
- Deployment as Domino Model API endpoints

## Configuration Requirements

Essential configuration sections in `monitoring_config.json`:
- `domino.base_url` - Domino instance URL
- `model_api.endpoint_url` - Deployed model API endpoint
- `model_api.token` - Model API authentication token
- `model_monitor.model_id` - Model Monitor ID from UI
- `data_sources.ground_truth` - Ground truth data source name

## Important Notes

- Training set names must be unique across the entire Domino instance
- Ground truth upload requires 1+ hour delay after prediction generation
- Feature names in training data must match prediction capture features
- Data source names must be consistent across Domino UI, project settings, and Model Monitor
- Quality metrics appear 24-48 hours after ground truth matching