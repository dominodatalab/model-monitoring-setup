# Model Monitor Extension

Streamlit dashboard for comprehensive model monitoring with custom metrics capabilities. Integrates with Domino's Model Monitoring API v2 and Training Sets API to compare training data against recent prediction data.

## Features

- **Custom Metrics**: Compute drift detection using Training Sets API and prediction data with pre-built statistical tests (KS, PSI, JS Divergence) or custom Python functions
- **Model Comparison Dashboard**: View and compare registered models with performance metrics and filtering
- **Smart Project Filtering**: Shows only models with prediction data in the current project
- **Flexible Time Windows**: Default 24-hour analysis with customizable date ranges

## Quick Deploy

1. **Navigate to Apps** in Domino UI → Create New App
2. **Configure:**
   - **File to Execute**: `app-model-monitor-extension/app.sh`
   - **Hardware Tier**: Select appropriate compute
   - **Environment**: Python with Streamlit support
   - **Thumbnail**: Download the `mnt/code/app-model-monitor-extension/app-images/app_thumbnail.png` and upload
3. **Publish** the app

📖 **Full guide**: [Domino Apps Documentation](https://docs.dominodatalab.com/en/cloud/user_guide/cd0095/publish-and-share-an-app/)


## Usage

1. **Custom Metrics** (default): Select models, training sets, and compute drift metrics
2. **Model Comparison Dashboard**: Overview of all monitored models

## Notes

- Only shows models registered in Model Monitor with available prediction data
- Must deploy app in the project containing the models you want to monitor
- Models need monitoring enabled and TrainingSet configured for drift detection