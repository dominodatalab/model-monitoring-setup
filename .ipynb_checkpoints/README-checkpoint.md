# Model Monitoring Setup Guide

Quick setup for Domino Model Monitor with automated prediction capture and ground truth tracking.

---

## Prerequisites

- Access to create or use a Domino data source
- Running in a Domino workspace (for automatic API key access)

**Note:** `DOMINO_USER_API_KEY` is automatically available in Domino workspaces - no manual configuration needed.

---

## Step 1: Register Training Set

Register your training data as a baseline for drift detection. **Note** that the training set must have a unique name across your Domino instance. 

```bash
python 1_register_training_set.py --file /path/to/training.csv --name "My Training Set"
```

The script accepts:
- **CSV files** (`.csv`)
- **Parquet files** (`.parquet`, `.pq`)
- Any pandas-readable format

**Note:** Feature names in your training data should match the features captured during prediction.

---

## Step 2: Create Model Deployment Script

### 2a. Export Model (Optional)

If you need to export a model from a registered Domino model:

```bash
python 2_optional_export_model.py --metric accuracy
python 2_optional_export_model.py --run-id YOUR_RUN_ID
```

### 2b. Add Prediction Capture

Modify your model deployment script to capture predictions. See `prediction_capture_example.py` for reference.

**Basic Setup:**

```python
from domino_data_capture.data_capture_client import DataCaptureClient
import uuid
from datetime import datetime, timezone

# Define feature and prediction names
feature_names = ['feature_1', 'feature_2', 'feature_3']
predict_names = ['predicted_class', 'confidence_score']

# Initialize client
data_capture_client = DataCaptureClient(feature_names, predict_names)

def predict(input_data):
    # Your prediction logic
    feature_values = [input_data['feature_1'], input_data['feature_2'], input_data['feature_3']]
    predicted_class = model.predict(feature_values)
    confidence = model.predict_proba(feature_values).max()

    # Capture prediction
    event_id = str(uuid.uuid4())
    event_time = datetime.now(timezone.utc).isoformat()

    data_capture_client.capturePrediction(
        feature_values,
        [predicted_class, confidence],
        event_id=event_id,
        timestamp=event_time
    )

    return {
        "predicted_class": predicted_class,
        "confidence_score": confidence,
        "event_id": event_id,
        "timestamp": event_time
    }
```

**Note:** Feature values must be explicitly provided to `capturePrediction()` in the order defined in `feature_names`.

### 2c. Deploy Model Endpoint

Deploy your model as a Domino Model API endpoint:

1. Navigate to your project in Domino
2. Go to **Deployments > Model APIs**
3. Click **New Model API**
4. Configure:
   - **File**: Your deployment script (e.g., `predict.py`)
   - **Function**: Your prediction function (e.g., `predict`)
   - **Environment**: Select appropriate environment
5. Click **Publish**

📖 **Details:** [Deploy from the UI](https://docs.dominodatalab.com/en/cloud/user_guide/8dbc91/deploy-domino-endpoints/#_deploy_from_the_ui)

---

## Step 3: Select Training Set

Once your model endpoint is running, enable drift detection by selecting the training set:

1. Navigate to your deployed model endpoint
2. Go to **Settings > Monitoring**
3. Select the training set you registered in Step 1
4. Click **Save**

This enables automatic drift detection based on your training data baseline.

📖 **Details:** [Endpoint Drift Detection](https://docs.dominodatalab.com/en/cloud/user_guide/c97091/domino-endpoint-drift-detection/#_select_the_training_set)

---

## Step 4: Configure Data Sources

You need to configure data sources in **three places**:

### 4a. Create Main Data Source

First, create a data source in Domino for storing ground truth data:

1. Navigate to **Data > Data Sources** in Domino UI
2. Click **New Data Source**
3. Select your storage type (S3, Azure Blob, GCS, etc.)
4. Configure connection settings and choose a descriptive name (e.g., `my-model-ground-truth`)

📖 **Details:** [Connect a Data Source](https://docs.dominodatalab.com/en/cloud/user_guide/8c7833/connect-a-data-source/)

### 4b. Connect to Project

Make the data source available in your project:

1. Go to **Project Settings > Data**
2. Click **Add Data Source**
3. Select your ground truth data source

📖 **Details:** [Use Data Sources](https://docs.dominodatalab.com/en/cloud/user_guide/fa5f3a/use-data-sources/)

### 4c. Configure in Model Monitor

Configure the same data source in Model Monitor for ground truth ingestion:

1. Navigate to **Model Monitor** in Domino UI
2. Select your model
3. Go to **Configuration > Ground Truth**
4. Add the data source you created in step 3a

**Important:** Use the **same name** for the data source in all three places (main data source, project connection, and Model Monitor) to avoid confusion.

---

## Step 5: Configure Monitoring

Run the interactive setup to configure monitoring:

```bash
python 3_setup_monitoring.py
```

You'll be prompted for:
1. **Domino base URL** - Your Domino instance URL
2. **Model API endpoint URL** - From your deployed model
3. **Model API token** - From model API settings
4. **Model Monitor ID** - From Model Monitor UI
5. **Ground truth data source name** - Name from Step 3
6. **Test data path** - Location of test data for predictions

The script creates `monitoring_config.json` with your settings.

---

## Step 6: Generate Predictions

Generate test predictions and upload ground truth:

```bash
python 4_generate_predictions.py --count 30
```

This will:
- Call your model API with test data
- Capture predictions automatically (via your deployment script)
- Upload ground truth to the data source

---

## Step 7: Upload Ground Truth Configuration

⏰ **IMPORTANT:** Wait at least 1 hour after generating predictions before running this step.

This allows ground truth data to be fully uploaded to S3.

```bash
python 5_upload_ground_truth.py
python 5_upload_ground_truth.py --hours 168  # Last 7 days
```

This registers the ground truth datasets with Model Monitor for quality metrics.

---

## Verification

1. **Check Predictions:** Navigate to Model Monitor UI → Model Details → Predictions
2. **Check Ground Truth:** Model Monitor UI → Ground Truth Status
3. **Check Metrics:** Quality metrics appear after ground truth is matched (24-48 hours)

---

## File Reference

- `1_register_training_set.py` - Register training baseline
- `prediction_capture_example.py` - Example code for deployment script
- `2_optional_export_model.py` - Export models from MLflow (optional)
- `3_setup_monitoring.py` - Interactive configuration
- `4_generate_predictions.py` - Generate test predictions
- `5_upload_ground_truth.py` - Register ground truth datasets
- `config_loader.py` - Configuration management utility

---

## Troubleshooting

**No quality metrics appearing:**
- Verify ground truth status is "Active" in Model Monitor UI
- Ensure `event_id` matches between predictions and ground truth
- Wait 24-48 hours for initial ingestion

**Configuration errors:**
- Run `python 3_setup_monitoring.py` to reconfigure
- Check data source name matches exactly

**Upload failures:**
- Verify data source is connected to project
- Check credentials and permissions
- Ensure Model Monitor ID is correct
