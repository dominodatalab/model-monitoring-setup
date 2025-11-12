# Model Monitoring Dashboard - User Guide

## Overview

This Streamlit app provides an interface for monitoring ML models in production, computing custom metrics, and comparing model performance.

## Navigation

The app has two main pages accessible from the sidebar:
- **Custom Metrics** - Compute and track custom monitoring metrics
- **Model Comparison Dashboard** - Compare models and view monitoring data

---

## Custom Metrics Page

### 1. Select Model
Choose the model you want to monitor from the dropdown. Only models with available prediction data are shown.

### 2. Choose Metric Type
Select from built-in statistical tests or create your own:
- **Kolmogorov-Smirnov Test** - Detects distribution changes between datasets
- **Population Stability Index (PSI)** - Measures distribution shift (banking/credit scoring)
- **Jensen-Shannon Divergence** - Symmetric measure of distribution similarity
- **Custom Python Function** - Write your own metric logic

### 3. Configure Data Sources

#### Baseline Data (Training Data)
- **Preferred**: Select a Training Set from the API (if available)
- **Fallback**: Use historical prediction data by selecting date range
- Click **Confirm Selection** if using a Training Set

#### Current Data (Recent Predictions)
- Select time period (7, 14, or 30 days)
- Uses most recent prediction data

### 4. Set Metric Parameters
Configure thresholds and parameters specific to your chosen metric type:
- **KS Test**: Significance level, threshold
- **PSI**: Number of bins, warning/critical thresholds
- **JS Divergence**: Threshold value

### 5. Configure Alerts (Optional)
Enable alerts to be notified when metrics breach thresholds:
- Choose condition: GREATER_THAN, LESS_THAN, BETWEEN, etc.
- Set threshold values
- Add custom alert message
- Add tags for organization (optional)

### 6. Compute Metric
Click **🚀 Compute Metric** to:
- Load and compare baseline vs. current data
- Calculate the metric value
- Log results to Domino Model Monitoring
- Send alerts (if configured)
- Display results with status indicators

### 7. Generate Scheduled Job Script
After computing a metric, click **📄 Generate Job Script** to:
- Create a Python script that automates this metric computation
- Start a Domino job to save the script to `artifacts/custom_metrics/`
- Receive instructions for scheduling the job

**To schedule the generated script:**
1. Wait for job completion
2. Navigate to **Jobs** in Domino
3. Create new job with file: `artifacts/custom_metrics/{metric_name}_job.py`
4. Set schedule (daily/weekly/hourly)
5. Test manually before scheduling

### 8. View Metric History
Switch to the **Metric History** tab to:
- Filter metrics by model and name
- Select date range
- View historical metric values in a table
- Visualize trends over time with interactive charts

---

## Model Comparison Dashboard

### Compare Models
1. Select **up to 3 models** from the multi-select dropdown
2. Click **Compare Selected Models** to view:
   - **Data Drift Summary**: Variables drifted, average drift scores
   - **Model Quality Summary**: Latest quality metrics, baseline scores
   - **Prediction Traffic**: Volume of predictions over time

### View All Models
Scroll down to see a comprehensive table of all registered models with:
- Model name and version
- Type (Classification/Regression)
- Status
- Scheduled checks
- Variables drifted
- Last check date

---

## Tips & Best Practices

### Custom Metrics
- **Test first**: Compute metrics manually before scheduling
- **Use Training Sets**: More accurate baseline than historical predictions
- **Set meaningful alerts**: Avoid alert fatigue with appropriate thresholds
- **Add tags**: Organize metrics for easier filtering

### Model Comparison
- **Limit to 3 models**: Keeps visualizations readable
- **Check drift regularly**: Set up scheduled checks for production models
- **Monitor traffic patterns**: Unusual traffic can indicate issues

### Troubleshooting
- **No models showing**: Ensure models are registered in Model Monitor and have prediction data
- **Job script failed**: Check job logs for Python/environment errors
- **Metric computation slow**: Large datasets may take time; consider sampling

---

## Environment Variables

The app uses these environment variables (configured automatically in Domino):
- `DOMINO_USER_API_KEY` - Your Domino API key
- `DOMINO_PROJECT_OWNER` - Project owner username
- `DOMINO_PROJECT_NAME` - Project name
- `DOMINO_PROJECT_ID` - Project ID

---

## Support & Documentation

- **Model Monitoring API Docs**: [Domino Model Monitoring](https://docs.dominodatalab.com/)
- **Training Sets API**: [Training Sets Guide](https://docs.dominodatalab.com/)
- **Scheduling Jobs**: [Jobs Documentation](https://docs.dominodatalab.com/en/cloud/user_guide/5dce1f/schedule-jobs/)

---

## Project Context

**Important**: This app only shows models associated with the current Domino project. Models from other projects are filtered out.
