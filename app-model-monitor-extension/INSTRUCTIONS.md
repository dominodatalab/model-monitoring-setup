# Model Monitoring Dashboard

Professional Streamlit dashboard for monitoring ML models via Domino Model Monitoring API

## Quick Start

### Configuration

Create a file called config.py based on the config_template.py and fill in your URL

Uses environment variables (set automatically in Domino):
- `DOMINO_USER_API_KEY` - Your API key 

### Run the Dashboard

**Deploy as Domino App:**
1. Navigate to **Deployments > Apps** in Domino UI
2. Click **Publish Domino App**
3. Configure app settings:
   - **Name**: Model Monitoring Dashboard
   - **Launch file**: `launch.sh` (or `app.py` for Streamlit)
   - **Environment**: Select environment with Streamlit
   - **Hardware**: Select appropriate compute tier
4. Set **access controls** (who can view the dashboard)
5. Click **Publish** to deploy

📖 **Full deployment guide**: [Publish and Share an App](https://docs.dominodatalab.com/en/cloud/user_guide/cd0095/publish-and-share-an-app/)

**Local development:** Run `streamlit run app.py` in a workspace. Generate the URL by filling in your Domino URL and running in a terminal: 

```bash
echo -e "import os\nprint('https://your-domino-url/{}/{}/notebookSession/{}/proxy/8501/'.format(os.environ['DOMINO_PROJECT_OWNER'], os.environ['DOMINO_PROJECT_NAME'], os.environ['DOMINO_RUN_ID']))" | python3
```

## Features

### Models Page
- **Compare up to 3 models** side-by-side
- **Drift Metrics**: Variables drifted, predictions analyzed
- **Quality Metrics**: Ground truth matches, accuracy metrics
- **Traffic**: Prediction volume over time
- **Date range selection** for filtering data
- **Expandable cards** for full model details

### Custom Metrics Page
- **Compute Metrics**: KS Test, PSI, JS Divergence, or custom functions
- **View History**: Time series of logged metrics
- **Mathematical formulas** and implementation code

## Important Notes

### What Models Are Shown
**Only models registered in Model Monitor** appear - not regular deployments.

Models must have:
1. Deployed as Domino endpoint
2. Registered with Model Monitoring
3. TrainingSet configured for drift detection
4. Drift/quality checks scheduled


## Troubleshooting

**No models found:**
- Ensure model has monitoring enabled
- Verify TrainingSet is registered
- Check drift/quality checks are scheduled

**API errors:**
- Verify `DOMINO_USER_API_KEY` is set
- Check you have Model Monitor permissions
- Confirm external URL is being used

**Refresh not working:**
- Use the 🔄 Refresh button (clears caches)
- Hard refresh browser (Ctrl+Shift+R)

## Files

- `app.py` - Streamlit dashboard
- `api_client.py` - Model Monitoring API v2 client
- `config.py` - Configuration loader
- `launch.sh` - Launch script
- `INSTRUCTIONS.md` - This file
