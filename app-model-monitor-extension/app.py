#!/usr/bin/env python3
"""
Domino Model Monitoring Dashboard

A professional Streamlit application for monitoring ML models in production.
Uses Model Monitoring API v2 to show only models registered in Model Monitor.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import List, Dict, Any
import sys
import time
from pathlib import Path

# Add current directory to path for imports
app_dir = Path(__file__).parent
sys.path.insert(0, str(app_dir))

# Change working directory to app directory to ensure relative imports work
import os
os.chdir(str(app_dir))

from api_client import ModelMonitoringClient
import config

# Try to import domino for custom metrics
try:
    import domino
    DOMINO_AVAILABLE = True
except ImportError:
    DOMINO_AVAILABLE = False

# Try to import training sets separately
try:
    from domino_data.training_sets.client import (
        list_training_sets,
        list_training_set_versions,
        get_training_set_version
    )
    TRAINING_SETS_AVAILABLE = True
except ImportError:
    TRAINING_SETS_AVAILABLE = False

# ==================== Page Configuration ====================

st.set_page_config(
    page_title="Model Monitoring Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Custom Styling (Domino Theme) ====================

DOMINO_STYLE = """
<style>
    /* Domino color palette */
    :root {
        --domino-dark-bg: #3C3A42;
        --domino-text: #97A3B7;
        --domino-accent: #626262;
        --domino-primary: #FF6B35;
        --domino-success: #4CAF50;
        --domino-warning: #FFA726;
        --domino-error: #EF5350;
    }

    /* Main container styling */
    .main {
        background-color: #0E1117;
    }

    /* Headers */
    h1 {
        color: #FFFFFF;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    h2, h3 {
        color: var(--domino-text);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-weight: 500;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: var(--domino-dark-bg);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: var(--domino-text);
    }

    /* Cards and containers */
    .stMetric {
        background-color: var(--domino-dark-bg);
        padding: 1rem;
        border-radius: 0.5rem;
    }

    /* Metric text colors */
    .stMetric label {
        color: var(--domino-text) !important;
    }

    .stMetric [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 600;
    }

    .stMetric [data-testid="stMetricDelta"] {
        color: var(--domino-text) !important;
    }

    /* Buttons */
    .stButton>button {
        background-color: var(--domino-primary);
        color: white;
        border: none;
        border-radius: 0.25rem;
        font-weight: 500;
    }

    .stButton>button:hover {
        background-color: #E55A2B;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 500;
    }

    .status-healthy {
        background-color: var(--domino-success);
        color: white;
    }

    .status-warning {
        background-color: var(--domino-warning);
        color: white;
    }

    .status-error {
        background-color: var(--domino-error);
        color: white;
    }
</style>
"""

st.markdown(DOMINO_STYLE, unsafe_allow_html=True)

# ==================== Initialize API Client ====================

@st.cache_resource(ttl=60)  # Cache for 1 minute only
def get_api_client(_cache_buster: str = "v3"):
    """Initialize and cache the API client."""
    try:
        return ModelMonitoringClient(
            api_host=config.API_HOST,
            api_key=config.API_KEY
        )
    except Exception as e:
        st.error(f"Failed to initialize API client: {e}")
        st.info("Please ensure DOMINO_USER_API_KEY is set in your environment")
        st.stop()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_training_sets_list():
    """Get and cache the list of training sets."""
    if not TRAINING_SETS_AVAILABLE:
        return []
    try:
        return list_training_sets()
    except Exception as e:
        st.error(f"Failed to get training sets: {e}")
        return []

# Use timestamp-based cache buster to force reload every minute
cache_version = f"v3_{int(time.time() // 60)}"  # Changes every minute
client = get_api_client(cache_version)

# ==================== Helper Functions ====================

def format_timestamp(ts: int) -> str:
    """Format Unix timestamp to readable format."""
    try:
        dt = datetime.fromtimestamp(ts)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return 'N/A'

def get_status_badge(status: str) -> str:
    """Generate HTML for status badge."""
    status_lower = status.lower() if status else 'unknown'

    if status_lower in ['created', 'active', 'healthy']:
        badge_class = 'status-healthy'
    elif status_lower in ['warning', 'degraded']:
        badge_class = 'status-warning'
    elif status_lower in ['error', 'failing', 'critical']:
        badge_class = 'status-error'
    else:
        badge_class = 'status-badge'

    return f'<span class="status-badge {badge_class}">{status}</span>'

def load_training_set_data(training_set_info: dict) -> pd.DataFrame:
    """
    Load data from a Domino Training Set.
    
    Args:
        training_set_info: Dictionary containing training set metadata from list_training_sets()
    
    Returns:
        DataFrame with training data
    """
    try:
        if not TRAINING_SETS_AVAILABLE:
            raise Exception("Training Sets API not available")
        
        training_set_name = training_set_info.get('name')
        if not training_set_name:
            raise Exception("Training set name not found")
        
        # Get the latest version of the training set
        versions = list_training_set_versions(training_set_name=training_set_name)
        if not versions:
            raise Exception(f"No versions found for training set '{training_set_name}'")
        
        # Get the latest version (highest number)
        latest_version = max(versions, key=lambda v: v.number)
        
        # Get the training set version object
        ts_version = get_training_set_version(training_set_name, latest_version.number)
        
        # Load the training data as pandas DataFrame
        df = ts_version.load_training_pandas()
        
        st.caption(f"✅ Loaded {len(df):,} training samples from training set '{training_set_name}' (version {latest_version.number})")
        return df
        
    except Exception as e:
        st.error(f"Error loading training set: {e}")
        return pd.DataFrame()

def load_prediction_data(start_date: datetime, end_date: datetime, model_id: str = None) -> pd.DataFrame:
    """
    Load prediction data for custom metric computation.

    This function attempts to load data in the following priority:
    1. Actual prediction data from /mnt/data/prediction_data (when model_id provided)
    2. Ground truth data from model-monitor-storage data source
    3. Training data from local filesystem as fallback

    Args:
        start_date: Start of date range
        end_date: End of date range
        model_id: Model monitor ID (optional, used to load specific model predictions)

    Returns:
        DataFrame with prediction data
    """
    try:
        # Priority 1: Try to load actual prediction data from /mnt/data/prediction_data
        if model_id:
            try:
                st.info(f"📊 Loading prediction data for model {model_id}")
                
                prediction_base = Path("/mnt/data/prediction_data")
                model_dir = prediction_base / model_id
                
                if model_dir.exists():
                    # Find all parquet files within the date range
                    all_files = []
                    
                    # Generate date range
                    current_date = start_date.date()
                    end_date_only = end_date.date()
                    
                    while current_date <= end_date_only:
                        date_str = current_date.strftime("%Y-%m-%d") + "Z"
                        date_dir = model_dir / f"$$date$$={date_str}"
                        
                        if date_dir.exists():
                            # Find all hour directories for this date
                            for hour_dir in date_dir.iterdir():
                                if hour_dir.is_dir() and hour_dir.name.startswith("$$hour$$="):
                                    # Find all parquet files in this hour
                                    for file_path in hour_dir.glob("*.parquet"):
                                        all_files.append(file_path)
                        
                        # Move to next date
                        current_date += pd.Timedelta(days=1).to_pytimedelta()
                    
                    if all_files:
                        # Load and combine all prediction files
                        dfs = []
                        for file_path in all_files:
                            try:
                                df_temp = pd.read_parquet(file_path)
                                dfs.append(df_temp)
                            except Exception as e:
                                st.warning(f"Could not load {file_path.name}: {e}")
                        
                        if dfs:
                            df = pd.concat(dfs, ignore_index=True)
                            
                            # Filter by timestamp if within range
                            if 'timestamp' in df.columns:
                                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                                start_date_tz = pd.Timestamp(start_date, tz='UTC')
                                end_date_tz = pd.Timestamp(end_date, tz='UTC')
                                df = df[(df['timestamp'] >= start_date_tz) & (df['timestamp'] <= end_date_tz)]
                            
                            # Extract only feature columns for drift analysis
                            feature_cols = [col for col in df.columns if 'feature' in col]
                            if feature_cols:
                                df_features = df[feature_cols]
                                st.caption(f"✅ Loaded {len(df_features):,} prediction records with {len(feature_cols)} features from {len(dfs)} file(s)")
                                return df_features
                            else:
                                st.warning("No feature columns found in prediction data")
                
                st.info("📊 No prediction data found for this model/date range, trying alternative sources")
                
            except Exception as e:
                st.warning(f"⚠️ Could not load prediction data: {e}, trying alternative sources")

        # Priority 2: Try to load ground truth data from data source
        try:
            from domino.data_sources import DataSourceClient
            import io

            st.info("📊 Loading ground truth data from model-monitor-storage")

            object_store = DataSourceClient().get_datasource('model-monitor-storage')
            objects = object_store.list_objects()

            # Find ground truth files within date range
            ground_truth_files = []
            for obj in objects:
                key = obj.key if hasattr(obj, 'key') else str(obj)
                if 'ground_truth' in key and key.endswith('.csv'):
                    # Filter by model_id if provided
                    if model_id and model_id not in key:
                        continue
                    ground_truth_files.append(key)

            if ground_truth_files:
                # Load and combine all ground truth files
                dfs = []
                for file_key in ground_truth_files:
                    try:
                        file_obj = io.BytesIO()
                        object_store.download_fileobj(file_key, file_obj)
                        file_obj.seek(0)
                        df_temp = pd.read_csv(file_obj)
                        dfs.append(df_temp)
                    except Exception as e:
                        st.warning(f"Could not load {file_key}: {e}")

                if dfs:
                    df = pd.concat(dfs, ignore_index=True)

                    # Filter by date range if timestamp column exists
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                        # Make start_date and end_date timezone-aware
                        start_date_tz = pd.Timestamp(start_date, tz='UTC')
                        end_date_tz = pd.Timestamp(end_date, tz='UTC')
                        df = df[(df['timestamp'] >= start_date_tz) & (df['timestamp'] <= end_date_tz)]

                    st.caption(f"✅ Loaded {len(df):,} ground truth records from {len(dfs)} file(s)")
                    return df

        except ImportError:
            st.warning("⚠️ DataSourceClient not available, falling back to training data")
        except Exception as e:
            st.warning(f"⚠️ Could not load from data source: {e}, falling back to training data")

        # Priority 3: Fallback to training data as proxy
        st.info("📊 Loading training data as proxy for prediction data")

        # Try multiple common training data file names and locations
        possible_paths = [
            Path('/mnt/data/transformed_cc_transactions.csv'),
            Path('/mnt/artifacts/transformed_cc_transactions.csv'),
            Path('/domino/datasets/local/Fraud-Detection-Workshop/transformed_cc_transactions.csv'),
            Path('/mnt/artifacts/training_data.csv'),
            Path('/mnt/data/training_data.csv'),
            Path('/mnt/code/training_data.csv'),
            Path('/mnt/artifacts/test_data.csv'),
            Path('/mnt/data/test_data.csv')
        ]

        data_path = None
        for path in possible_paths:
            if path.exists():
                data_path = path
                break

        if data_path is None:
            raise FileNotFoundError("Training data not found. Cannot simulate prediction data.")

        df = pd.read_csv(data_path)

        # Sample data based on date range to simulate different time periods
        # Use date as seed for reproducibility within the same date range
        sample_size = min(1000, len(df))
        random_seed = int(start_date.timestamp()) % 10000
        sampled_df = df.sample(n=sample_size, random_state=random_seed)

        st.caption(f"⚠️ Using {len(sampled_df):,} training samples from {data_path.name} as proxy (seed: {random_seed})")

        return sampled_df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# ==================== Sidebar Navigation ====================

st.sidebar.title("📊 Model Monitoring")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Custom Metrics", "Model Comparison Dashboard"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "Monitor production ML models with drift detection, "
    "quality metrics, and custom monitoring capabilities."
)

# ==================== PAGE 1: Models (List + Comparison) ====================

if page == "Model Comparison Dashboard":
    st.title("Model Comparison Dashboard")
    st.markdown("View and compare models registered in Model Monitoring")

    try:
        # Fetch models with more checks for date filtering
        with st.spinner("Loading models from Model Monitor..."):
            response = client.list_models(page_size=100, number_of_last_checks=100)  # Increased from 10 to 100
            models = response.get('modelDashboardItems', [])

        if not models:
            st.warning("No models found in Model Monitor. Deploy a model with monitoring enabled.")
        else:
            # Display summary stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Models", len(models))
            with col2:
                classification_count = response.get('classificationModelCount', 0)
                st.metric("Classification", classification_count)
            with col3:
                regression_count = response.get('regressionModelCount', 0)
                st.metric("Regression", regression_count)
            with col4:
                drift_scheduled = sum(1 for m in models if m.get('isDataDriftCheckScheduled'))
                st.metric("Drift Monitoring", drift_scheduled)

            st.markdown("---")

            # Model selection for comparison
            st.markdown("### 📊 Compare Models (Select up to 3)")

            model_options = {
                f"{m.get('name', 'Unnamed')} v{m.get('version', 'N/A')} ({m.get('modelType', 'unknown')})": m.get('id')
                for m in models
            }

            selected_model_names = st.multiselect(
                "Select models to compare",
                options=list(model_options.keys()),
                max_selections=3,
                help="Choose up to 3 models for detailed comparison"
            )

            # Show comparison if models are selected
            if selected_model_names:
                selected_model_ids = [model_options[name] for name in selected_model_names]

                st.markdown("---")
                col_header1, col_header2 = st.columns([4, 1])
                with col_header1:
                    st.markdown(f"### Comparing {len(selected_model_names)} Models")
                with col_header2:
                    if st.button("🔄 Refresh Data", key="refresh_comparison"):
                        st.cache_data.clear()
                        st.cache_resource.clear()
                        st.rerun()

                # Date range selection
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input(
                        "Start Date",
                        value=datetime.now() - timedelta(days=30)
                    )
                with col2:
                    end_date = st.date_input(
                        "End Date",
                        value=datetime.now()
                    )

                # Convert dates to Unix timestamps for API calls
                start_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
                end_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())

                # Convert dates to ISO format for drift/quality APIs
                start_iso = datetime.combine(start_date, datetime.min.time()).isoformat() + 'Z'
                end_iso = datetime.combine(end_date, datetime.max.time()).isoformat() + 'Z'

                # Comparison tabs
                tab1, tab2, tab3 = st.tabs(["📈 Drift Metrics", "🎯 Quality Metrics", "📊 Traffic"])

                with tab1:
                    st.markdown("#### Data Drift Summary")

                    drift_data = []
                    for model_id, model_name in zip(selected_model_ids, selected_model_names):
                        # Find the model details
                        model_detail = next((m for m in models if m['id'] == model_id), None)

                        if model_detail:
                            # Get all drift checks and filter by date range
                            all_drift_checks = model_detail.get('dataDriftChecks', [])

                            # Filter checks within the selected date range
                            filtered_checks = [
                                check for check in all_drift_checks
                                if start_timestamp <= check.get('checkedOn', 0) <= end_timestamp
                            ]

                            if filtered_checks:
                                # Aggregate across all checks in the date range
                                total_predictions = sum(check.get('numberOfPredictions', 0) for check in filtered_checks)

                                # Get most recent check for variables info
                                latest_check = filtered_checks[0]

                                drift_data.append({
                                    'Model': model_name.split(' (')[0],  # Clean name
                                    'Variables Monitored': latest_check.get('totalNumberOfVariables', 0),
                                    'Variables Drifted': len(latest_check.get('variablesDrifted', [])),
                                    'Predictions': total_predictions,
                                    'Checks in Range': len(filtered_checks),
                                    'Last Checked': format_timestamp(latest_check.get('checkedOn', 0))
                                })
                            else:
                                drift_data.append({
                                    'Model': model_name.split(' (')[0],
                                    'Variables Monitored': 0,
                                    'Variables Drifted': 0,
                                    'Predictions': 0,
                                    'Checks in Range': 0,
                                    'Last Checked': 'No data in range'
                                })

                    if drift_data:
                        df_drift = pd.DataFrame(drift_data)

                        # Create visualizations
                        col1, col2 = st.columns(2)

                        with col1:
                            fig1 = px.bar(
                                df_drift,
                                x='Model',
                                y='Variables Drifted',
                                title='Variables with Drift Detected',
                                template='plotly_dark',
                                height=350,
                                color_discrete_sequence=['#FF6B35']
                            )
                            st.plotly_chart(fig1, use_container_width=True)

                        with col2:
                            fig2 = px.bar(
                                df_drift,
                                x='Model',
                                y='Predictions',
                                title='Predictions Analyzed',
                                template='plotly_dark',
                                height=350,
                                color_discrete_sequence=['#4CAF50']
                            )
                            st.plotly_chart(fig2, use_container_width=True)

                        # Display table
                        st.dataframe(df_drift, use_container_width=True)

                with tab2:
                    st.markdown("#### Model Quality Summary")

                    quality_data = []
                    quality_metrics_aggregated = {}

                    for model_id, model_name in zip(selected_model_ids, selected_model_names):
                        model_detail = next((m for m in models if m['id'] == model_id), None)

                        if model_detail:
                            # Get all quality checks and filter by date range
                            all_quality_checks = model_detail.get('modelQualityChecks', [])

                            # Filter checks within the selected date range
                            filtered_checks = [
                                check for check in all_quality_checks
                                if start_timestamp <= check.get('checkedOn', 0) <= end_timestamp
                            ]

                            if filtered_checks:
                                # Aggregate across ALL checks in the date range
                                total_matched = sum(check.get('matchedRowCount', 0) for check in filtered_checks)

                                # Collect ALL metrics from ALL checks
                                all_metrics = []
                                metrics_failure_count = {}

                                for check in filtered_checks:
                                    check_metrics = check.get('metrics', [])
                                    for metric in check_metrics:
                                        metric_name = metric.get('name', 'Unknown')
                                        is_failed = metric.get('isFailed', False)

                                        # Track failure count for each metric
                                        if metric_name not in metrics_failure_count:
                                            metrics_failure_count[metric_name] = {'failed': 0, 'total': 0, 'latest_value': metric.get('value', 0)}

                                        metrics_failure_count[metric_name]['total'] += 1
                                        metrics_failure_count[metric_name]['latest_value'] = metric.get('value', 0)  # Keep latest value

                                        if is_failed:
                                            metrics_failure_count[metric_name]['failed'] += 1

                                # Count total metrics that failed at least once
                                metrics_out_of_range = sum(1 for m in metrics_failure_count.values() if m['failed'] > 0)

                                # Get the most recent check timestamp
                                latest_check_time = max(check.get('checkedOn', 0) for check in filtered_checks)

                                quality_data.append({
                                    'Model': model_name.split(' (')[0],
                                    'Matched Rows': total_matched,
                                    'Checks in Range': len(filtered_checks),
                                    'Metrics Tracked': len(metrics_failure_count),
                                    'Metrics Out of Range': metrics_out_of_range,
                                    'Last Checked': format_timestamp(latest_check_time)
                                })

                                # Store aggregated metrics for detailed display
                                if metrics_failure_count:
                                    quality_metrics_aggregated[model_name.split(' (')[0]] = metrics_failure_count
                            else:
                                quality_data.append({
                                    'Model': model_name.split(' (')[0],
                                    'Matched Rows': 0,
                                    'Checks in Range': 0,
                                    'Metrics Tracked': 0,
                                    'Metrics Out of Range': 0,
                                    'Last Checked': 'No data in range'
                                })

                    if quality_data:
                        df_quality = pd.DataFrame(quality_data)

                        # Create visualizations
                        col1, col2 = st.columns(2)

                        with col1:
                            fig1 = px.bar(
                                df_quality,
                                x='Model',
                                y='Matched Rows',
                                title='Ground Truth Labels Matched',
                                template='plotly_dark',
                                height=350,
                                color_discrete_sequence=['#4CAF50']
                            )
                            st.plotly_chart(fig1, use_container_width=True)

                        with col2:
                            fig2 = px.bar(
                                df_quality,
                                x='Model',
                                y='Metrics Out of Range',
                                title='Metrics Out of Range',
                                template='plotly_dark',
                                height=350,
                                color_discrete_sequence=['#EF5350']
                            )
                            st.plotly_chart(fig2, use_container_width=True)

                        # Display table
                        st.dataframe(df_quality, use_container_width=True)

                        # Display detailed metrics if available
                        if quality_metrics_aggregated:
                            st.markdown("#### Detailed Quality Metrics (Across All Checks)")
                            for model_name, metrics_info in quality_metrics_aggregated.items():
                                st.markdown(f"**{model_name}:**")

                                # Show metrics in columns
                                metric_items = list(metrics_info.items())
                                cols = st.columns(min(len(metric_items), 4))

                                for i, (metric_name, info) in enumerate(metric_items):
                                    with cols[i % 4]:
                                        metric_display_name = metric_name.replace('_', ' ').title()
                                        latest_value = info['latest_value']
                                        failed_count = info['failed']
                                        total_count = info['total']

                                        # Show metric with failure rate
                                        if failed_count > 0:
                                            failure_rate = f"{failed_count}/{total_count} checks"
                                            st.metric(
                                                metric_display_name,
                                                f"{latest_value:.4f}",
                                                delta=f"Failed: {failure_rate}",
                                                delta_color="inverse"
                                            )
                                        else:
                                            st.metric(
                                                metric_display_name,
                                                f"{latest_value:.4f}",
                                                delta=f"✓ {total_count} checks",
                                                delta_color="normal"
                                            )

                                st.markdown("---")
                        else:
                            st.info("No quality metrics computed yet. Ensure ground truth data is registered and quality checks are scheduled.")

                with tab3:
                    st.markdown("#### Prediction Traffic")

                    # Debug info (collapsible)
                    with st.expander("🔍 Debug Info", expanded=False):
                        st.code(f"""
Date Range: {start_date} to {end_date}
Start Timestamp: {start_timestamp} ({datetime.fromtimestamp(start_timestamp)})
End Timestamp: {end_timestamp} ({datetime.fromtimestamp(end_timestamp)})
Cache Version: {cache_version}
                        """)

                    traffic_data = []
                    for model_id, model_name in zip(selected_model_ids, selected_model_names):
                        try:
                            # API returns array of traffic data per model
                            traffic_response = client.get_prediction_traffic(
                                model_id=model_id,
                                start_timestamp=start_timestamp,
                                end_timestamp=end_timestamp
                            )

                            # Sum up all queries from the traffic array
                            total_predictions = 0
                            if isinstance(traffic_response, list) and len(traffic_response) > 0:
                                model_traffic = traffic_response[0]
                                traffic_points = model_traffic.get('traffic', [])
                                total_predictions = sum(point.get('queries', 0) for point in traffic_points)

                            traffic_data.append({
                                'Model': model_name.split(' (')[0],
                                'Total Predictions': total_predictions
                            })
                        except Exception as e:
                            st.error(f"❌ Could not fetch traffic for {model_name}")
                            st.code(f"Error: {str(e)}\nModel ID: {model_id}")

                    if traffic_data:
                        df_traffic = pd.DataFrame(traffic_data)

                        fig = px.bar(
                            df_traffic,
                            x='Model',
                            y='Total Predictions',
                            title='Prediction Volume',
                            template='plotly_dark',
                            height=350,
                            color_discrete_sequence=['#FF6B35']
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        st.dataframe(df_traffic, use_container_width=True)
                    else:
                        st.info("No traffic data available for the selected date range.")

            # Model list section
            st.markdown("---")
            st.markdown("### 📋 All Models")

            # Display models in a table
            for model in models:
                model_id = model.get('id', 'N/A')
                model_name = model.get('name', 'Unnamed Model')
                model_version = model.get('version', 'N/A')
                model_type = model.get('modelType', 'unknown').title()
                model_status = model.get('modelStatus', 'unknown')
                created_at = format_timestamp(model.get('createdAt', 0))

                # Drift info
                drift_checks = model.get('dataDriftChecks', [])
                drift_scheduled = model.get('isDataDriftCheckScheduled', False)
                latest_drift = drift_checks[0] if drift_checks else {}
                variables_drifted = len(latest_drift.get('variablesDrifted', []))

                # Quality info
                quality_scheduled = model.get('isModelQualityCheckScheduled', False)

                # Create expandable section for each model
                with st.expander(f"**{model_name}** v{model_version} - {model_type}", expanded=False):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown("**Model Information**")
                        st.markdown(f"**ID:** `{model_id}`")
                        st.markdown(f"**Type:** {model_type}")
                        st.markdown(f"**Status:** {get_status_badge(model_status)}", unsafe_allow_html=True)
                        st.markdown(f"**Created:** {created_at}")

                    with col2:
                        st.markdown("**Monitoring Status**")
                        st.markdown(f"**Drift Scheduled:** {'✅' if drift_scheduled else '❌'}")
                        st.markdown(f"**Quality Scheduled:** {'✅' if quality_scheduled else '❌'}")
                        if variables_drifted > 0:
                            st.markdown(f"**Variables Drifted:** {variables_drifted}")

                    with col3:
                        st.markdown("**Latest Checks**")
                        if latest_drift:
                            st.markdown(f"**Predictions:** {latest_drift.get('numberOfPredictions', 0)}")
                            st.markdown(f"**Variables:** {latest_drift.get('totalNumberOfVariables', 0)}")
                            st.markdown(f"**Checked:** {format_timestamp(latest_drift.get('checkedOn', 0))}")

                    # Show full details
                    if st.checkbox("Show full details", key=f"full_{model_id}"):
                        st.json(model)

    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.exception(e)

# ==================== PAGE 2: Custom Metrics ====================

elif page == "Custom Metrics":
    st.title("📊 Custom Metrics")
    st.markdown("Define and compute custom monitoring metrics for your models.")
    
    # Add project-specific note
    st.info(f"📁 **Project Context**: Only models associated with project `{config.DOMINO_PROJECT_OWNER}/{config.DOMINO_PROJECT_NAME}` (ID: `{config.DOMINO_PROJECT_ID}`) are available for custom metrics.")

    try:
        # Fetch available models and filter by prediction data correlation
        with st.spinner("Loading and filtering models..."):
            response = client.list_models()
            all_models = response.get('modelDashboardItems', [])
            
            # Get available prediction data model IDs (these are workbench model version IDs)
            prediction_base = Path("/mnt/data/prediction_data")
            prediction_model_version_ids = []
            if prediction_base.exists():
                prediction_model_version_ids = [d.name for d in prediction_base.iterdir() if d.is_dir()]
            
            # Filter models by matching workbenchModelVersionId with prediction data IDs
            # Need to call get_model() for each model to get detailed sourceDetails
            models = []
            for model in all_models:
                try:
                    # Get detailed model information which includes sourceDetails
                    detailed_model = client.get_model(model.get('id'))
                    source_details = detailed_model.get('sourceDetails', {})
                    workbench_model_version_id = source_details.get('workbenchModelVersionId')
                    
                    if workbench_model_version_id and workbench_model_version_id in prediction_model_version_ids:
                        # Use the detailed model info instead of the basic model info
                        models.append(detailed_model)
                except Exception as e:
                    # If we can't get detailed info, skip this model
                    st.warning(f"Could not get details for model {model.get('name', 'Unknown')}: {e}")
                    continue
            
            if len(models) < len(all_models):
                filtered_count = len(all_models) - len(models)
                st.caption(f"ℹ️ Filtered out {filtered_count} model(s) that are visible to the user but not part of this project. You must deploy the app in the project containing the models you would like to register metrics for")
                
                # Show debug info in expander
                with st.expander("🔍 Debug: Model Filtering Details", expanded=False):
                    st.markdown("**Available Models (with matching prediction data):**")
                    for model in models:
                        source_details = model.get('sourceDetails', {})
                        workbench_version_id = source_details.get('workbenchModelVersionId', 'N/A')
                        st.markdown(f"• {model.get('name')} v{model.get('version')} (Monitor ID: `{model.get('id')}`, Workbench Version ID: `{workbench_version_id}`)")
                    
                    st.markdown("**Filtered Models (no matching prediction data):**")
                    filtered_models = []
                    for model in all_models:
                        source_details = model.get('sourceDetails', {})
                        workbench_model_version_id = source_details.get('workbenchModelVersionId')
                        if not workbench_model_version_id or workbench_model_version_id not in prediction_model_version_ids:
                            filtered_models.append(model)
                    
                    for model in filtered_models:
                        source_details = model.get('sourceDetails', {})
                        workbench_version_id = source_details.get('workbenchModelVersionId', 'None')
                        st.markdown(f"• {model.get('name')} v{model.get('version')} (Monitor ID: `{model.get('id')}`, Workbench Version ID: `{workbench_version_id}`)")
                    
                    st.markdown("**Available Prediction Data IDs (Workbench Model Version IDs):**")
                    st.code(", ".join(prediction_model_version_ids))

        if not models:
            st.warning("No models with prediction data available for this project. Deploy a model with monitoring enabled and ensure prediction data is being captured.")
        else:
            # Create tabs for Compute and History
            tab1, tab2 = st.tabs(["🔬 Compute Metric", "📈 Metric History"])

            with tab1:
                st.markdown("### Define Custom Metric")

                # Model Selection Section
                st.markdown("#### 1️⃣ Select Model")
                model_options = {
                    f"{m.get('name', 'Unnamed')} v{m.get('version', 'N/A')}": m.get('id')
                    for m in models
                }

                selected_model_name = st.selectbox(
                    "Choose a model to analyze",
                    options=list(model_options.keys()),
                    help="Select the model you want to compute custom metrics for"
                )
                selected_model_id = model_options[selected_model_name] if selected_model_name else None

                st.markdown("---")

                # Metric Type Selection
                st.markdown("#### 2️⃣ Select Metric Type")

                col1, col2 = st.columns([2, 3])

                with col1:
                    metric_type = st.selectbox(
                        "Metric",
                        options=[
                            "Kolmogorov-Smirnov Test",
                            "Population Stability Index (PSI)",
                            "Jensen-Shannon Divergence",
                            "Custom Python Function"
                        ],
                        help="Select the type of custom metric to compute"
                    )
                    st.caption("⚠️ Please review the implementation code below to ensure it meets your requirements.")

                with col2:
                    # Show description based on selected metric
                    if metric_type == "Kolmogorov-Smirnov Test":
                        st.info("**KS Test**: Compares the distribution of prediction data between two time periods to detect drift.")
                    elif metric_type == "Population Stability Index (PSI)":
                        st.info("**PSI**: Measures the shift in population distribution between baseline and current data.")
                    elif metric_type == "Jensen-Shannon Divergence":
                        st.info("**JS Divergence**: Symmetric measure of similarity between two probability distributions.")
                    else:
                        st.info("**Python Function**: Define your own metric computation logic.")

                # Show mathematical formula
                st.markdown("#### 📐 Mathematical Formula")

                if metric_type == "Kolmogorov-Smirnov Test":
                    st.latex(r"D_{KS} = \max_{x} |F_{\text{baseline}}(x) - F_{\text{current}}(x)|")
                    st.markdown("""
                    Where:
                    - $D_{KS}$ = KS statistic (maximum distance between CDFs)
                    - $F_{\\text{baseline}}(x)$ = Cumulative distribution function of baseline data
                    - $F_{\\text{current}}(x)$ = Cumulative distribution function of current data
                    - Range: [0, 1], where 0 = identical distributions
                    """)

                elif metric_type == "Population Stability Index (PSI)":
                    st.latex(r"PSI = \sum_{i=1}^{n} (P_{\text{current},i} - P_{\text{baseline},i}) \times \ln\left(\frac{P_{\text{current},i}}{P_{\text{baseline},i}}\right)")
                    st.markdown("""
                    Where:
                    - $P_{\\text{current},i}$ = Proportion of current data in bin $i$
                    - $P_{\\text{baseline},i}$ = Proportion of baseline data in bin $i$
                    - $n$ = Number of bins
                    - Interpretation: PSI < 0.1 (stable), 0.1-0.25 (moderate drift), > 0.25 (severe drift)
                    """)

                elif metric_type == "Jensen-Shannon Divergence":
                    st.latex(r"JSD(P \| Q) = \frac{1}{2} D_{KL}(P \| M) + \frac{1}{2} D_{KL}(Q \| M)")
                    st.latex(r"M = \frac{1}{2}(P + Q)")
                    st.markdown("""
                    Where:
                    - $D_{KL}$ = Kullback-Leibler divergence
                    - $P$ = Baseline distribution
                    - $Q$ = Current distribution
                    - $M$ = Average of P and Q
                    - Range: [0, 1], where 0 = identical distributions
                    """)

                elif metric_type == "Custom Python Function":
                    st.markdown("Custom Python functions allow full flexibility in metric computation.")

                st.markdown("---")

                # Configuration Section
                st.markdown("#### 3️⃣ Configure Parameters")

                # Training Set Selection and Current Period
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Baseline Data** (Training Set)")
                    
                    # Training Set Selection UI will be added here
                    if not TRAINING_SETS_AVAILABLE:
                        st.warning("⚠️ Training Sets API not available. Using fallback date selection.")
                        st.info("💡 **Tip:** Install `domino-data` package to enable training set selection: `pip install domino-data`")
                        baseline_start = st.date_input(
                            "Start Date",
                            value=datetime.now() - timedelta(days=60),
                            key="baseline_start"
                        )
                        baseline_end = st.date_input(
                            "End Date",
                            value=datetime.now() - timedelta(days=30),
                            key="baseline_end"
                        )
                        selected_training_set = None
                        training_set_confirmed = False
                    else:
                        # Training set selection
                        try:
                            with st.spinner("Loading training sets..."):
                                # Get available training sets and order by most recent
                                training_sets = get_training_sets_list()
                                
                            if training_sets:
                                # For now, just sort by name (creation timestamp not available in TrainingSet object)
                                training_sets_sorted = sorted(training_sets, key=lambda x: x.name)
                                
                                # Create display options with metadata
                                training_set_options = {}
                                for ts in training_sets_sorted:
                                    # Get metadata from the meta dict if available
                                    total_records = ts.meta.get('total_records', 'Unknown')
                                    features = ts.meta.get('features', 'Unknown')
                                    model_type = ts.meta.get('model_type', 'Unknown')
                                    
                                    display_name = f"{ts.name} ({total_records} records, {features} features, {model_type})"
                                    # Convert TrainingSet to dict for easier handling
                                    training_set_options[display_name] = {
                                        'name': ts.name,
                                        'project_id': ts.project_id,
                                        'description': str(ts.description) if hasattr(ts.description, '__str__') else 'No description',
                                        'meta': ts.meta
                                    }
                                
                                # Training set selection dropdown
                                selected_training_set_name = st.selectbox(
                                    "Select Training Set",
                                    options=list(training_set_options.keys()),
                                    help="Training sets ordered by most recent creation date",
                                    key="training_set_selection"
                                )
                                
                                selected_training_set = training_set_options[selected_training_set_name]
                                
                                # Show training set details and confirmation
                                with st.expander("📋 Training Set Details", expanded=True):
                                    col_a, col_b = st.columns(2)
                                    with col_a:
                                        st.markdown(f"**Name:** {selected_training_set.get('name', 'N/A')}")
                                        st.markdown(f"**Description:** {selected_training_set.get('description', 'No description')}")
                                        st.markdown(f"**Project ID:** {selected_training_set.get('project_id', 'N/A')}")
                                    with col_b:
                                        meta = selected_training_set.get('meta', {})
                                        if meta.get('total_records'):
                                            st.markdown(f"**Total Records:** {meta.get('total_records')}")
                                        if meta.get('features'):
                                            st.markdown(f"**Features:** {meta.get('features')}")
                                        if meta.get('model_type'):
                                            st.markdown(f"**Model Type:** {meta.get('model_type')}")
                                    
                                    # Show additional metadata if available
                                    if meta:
                                        st.markdown("**Additional Metadata:**")
                                        for key, value in meta.items():
                                            if key not in ['total_records', 'features', 'model_type']:
                                                st.caption(f"• {key}: {value}")
                                    
                                    # Confirmation checkbox
                                    training_set_confirmed = st.checkbox(
                                        f"✅ Use '{selected_training_set.get('name')}' as baseline data",
                                        key="training_set_confirmed",
                                        help="Check this box to confirm using this training set as baseline"
                                    )
                                    
                                    if not training_set_confirmed:
                                        st.warning("⚠️ Please confirm the training set selection above to proceed")
                                
                            else:
                                st.warning("⚠️ No training sets found. Using fallback date selection.")
                                selected_training_set = None
                                training_set_confirmed = False
                                
                        except Exception as e:
                            st.error(f"❌ Failed to load training sets: {e}")
                            selected_training_set = None
                            training_set_confirmed = False
                        
                        # Fallback dates for backward compatibility
                        baseline_start = datetime.now() - timedelta(days=60)
                        baseline_end = datetime.now() - timedelta(days=30)

                with col2:
                    st.markdown("**Current Period** (Recent Predictions)")
                    
                    # Time period selection
                    time_period_days = st.number_input(
                        "Days of recent data",
                        min_value=1,
                        max_value=30,
                        value=1,  # Default to last 24 hours
                        help="Number of days of recent prediction data to load (default: 1 day = last 24 hours)",
                        key="time_period_days"
                    )
                    
                    # Calculate dates based on time period
                    current_end = datetime.now()
                    current_start = current_end - timedelta(days=time_period_days)
                    
                    # Show the calculated date range
                    st.caption(f"📅 Loading data from {current_start.strftime('%Y-%m-%d %H:%M')} to {current_end.strftime('%Y-%m-%d %H:%M')}")
                    
                    # Optional: Allow manual override
                    with st.expander("🛠️ Advanced: Manual Date Override", expanded=False):
                        manual_override = st.checkbox("Use custom date range", key="manual_override")
                        if manual_override:
                            current_start = datetime.combine(
                                st.date_input(
                                    "Start Date",
                                    value=current_start.date(),
                                    key="manual_current_start"
                                ),
                                datetime.min.time()
                            )
                            current_end = datetime.combine(
                                st.date_input(
                                    "End Date", 
                                    value=current_end.date(),
                                    key="manual_current_end"
                                ),
                                datetime.max.time()
                            )

                # Metric-specific parameters
                st.markdown("**Metric Parameters**")

                if metric_type == "Kolmogorov-Smirnov Test":
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        ks_threshold = st.number_input(
                            "Threshold",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.05,
                            step=0.01,
                            help="KS statistic threshold for detecting drift (typically 0.05-0.10)"
                        )
                    with col2:
                        significance_level = st.selectbox(
                            "Significance Level",
                            options=[0.01, 0.05, 0.10],
                            index=1,
                            help="Statistical significance level (alpha)"
                        )
                    with col3:
                        feature_selection = st.selectbox(
                            "Features",
                            options=["All Features", "Select Specific Features", "Top N Drifted"],
                            help="Which features to include in the analysis"
                        )

                    if feature_selection == "Select Specific Features":
                        st.multiselect(
                            "Choose features",
                            options=["Feature will be loaded from model schema"],
                            help="Select specific features to analyze"
                        )
                    elif feature_selection == "Top N Drifted":
                        st.number_input(
                            "Number of features",
                            min_value=1,
                            max_value=50,
                            value=10,
                            help="Analyze the top N features with highest drift"
                        )

                    # Show implementation
                    st.markdown("**Implementation Code** (read-only)")
                    st.code("""def compute_metric(baseline_data, current_data):
    \"\"\"
    Compute Kolmogorov-Smirnov test statistic for drift detection.

    Returns KS statistic and p-value for each feature.
    \"\"\"
    from scipy import stats
    import numpy as np

    results = []
    for column in baseline_data.columns:
        # Perform two-sample KS test
        ks_stat, p_value = stats.ks_2samp(
            baseline_data[column].dropna(),
            current_data[column].dropna()
        )

        results.append({
            'feature': column,
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'drifted': ks_stat > threshold or p_value < significance_level
        })

    # Aggregate results
    drifted_features = [r for r in results if r['drifted']]
    max_ks = max([r['ks_statistic'] for r in results])

    return {
        'value': float(max_ks),
        'metadata': {
            'total_features': len(results),
            'drifted_features': len(drifted_features),
            'feature_results': results,
            'baseline_samples': len(baseline_data),
            'current_samples': len(current_data)
        },
        'status': 'critical' if len(drifted_features) > len(results) * 0.3
                  else 'warning' if len(drifted_features) > 0
                  else 'ok'
    }""", language="python")

                elif metric_type == "Population Stability Index (PSI)":
                    col1, col2 = st.columns(2)
                    with col1:
                        psi_threshold = st.number_input(
                            "Warning Threshold",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.10,
                            step=0.01,
                            help="PSI > 0.10 indicates moderate drift"
                        )
                    with col2:
                        psi_critical = st.number_input(
                            "Critical Threshold",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.25,
                            step=0.01,
                            help="PSI > 0.25 indicates severe drift"
                        )

                    num_bins = st.slider(
                        "Number of bins for discretization",
                        min_value=5,
                        max_value=20,
                        value=10,
                        help="More bins = higher resolution, but requires more data"
                    )

                    # Show implementation
                    st.markdown("**Implementation Code** (read-only)")
                    st.code("""def compute_metric(baseline_data, current_data):
    \"\"\"
    Compute Population Stability Index (PSI) for drift detection.

    PSI measures the shift in population distribution.
    \"\"\"
    import numpy as np
    import pandas as pd

    def calculate_psi(baseline, current, bins=10):
        \"\"\"Calculate PSI for a single feature.\"\"\"
        # Create bins based on baseline distribution
        breakpoints = np.linspace(
            baseline.min(), baseline.max(), bins + 1
        )
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        # Bin both distributions
        baseline_binned = pd.cut(baseline, bins=breakpoints)
        current_binned = pd.cut(current, bins=breakpoints)

        # Calculate proportions
        baseline_counts = baseline_binned.value_counts().sort_index()
        current_counts = current_binned.value_counts().sort_index()

        baseline_props = baseline_counts / len(baseline)
        current_props = current_counts / len(current)

        # Replace zeros with small value to avoid log(0)
        baseline_props = baseline_props.replace(0, 0.0001)
        current_props = current_props.replace(0, 0.0001)

        # Calculate PSI
        psi = np.sum((current_props - baseline_props) *
                     np.log(current_props / baseline_props))

        return psi

    # Calculate PSI for each feature
    psi_results = {}
    for column in baseline_data.columns:
        psi_value = calculate_psi(
            baseline_data[column].dropna(),
            current_data[column].dropna(),
            bins=num_bins
        )
        psi_results[column] = psi_value

    # Aggregate
    max_psi = max(psi_results.values())
    avg_psi = np.mean(list(psi_results.values()))

    return {
        'value': float(max_psi),
        'metadata': {
            'average_psi': float(avg_psi),
            'feature_psi': psi_results,
            'baseline_samples': len(baseline_data),
            'current_samples': len(current_data)
        },
        'status': 'critical' if max_psi > psi_critical
                  else 'warning' if max_psi > psi_threshold
                  else 'ok'
    }""", language="python")

                elif metric_type == "Jensen-Shannon Divergence":
                    js_threshold = st.number_input(
                        "Threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.10,
                        step=0.01,
                        help="JS Divergence threshold (0 = identical, 1 = completely different)"
                    )

                    # Show implementation
                    st.markdown("**Implementation Code** (read-only)")
                    st.code("""def compute_metric(baseline_data, current_data):
    \"\"\"
    Compute Jensen-Shannon Divergence for drift detection.

    JS Divergence is a symmetric measure of distribution similarity.
    \"\"\"
    import numpy as np
    from scipy.spatial import distance
    from scipy.stats import entropy

    def calculate_js_divergence(baseline, current, bins=50):
        \"\"\"Calculate JS Divergence for a single feature.\"\"\"
        # Create histograms with same bins
        min_val = min(baseline.min(), current.min())
        max_val = max(baseline.max(), current.max())
        bins_range = np.linspace(min_val, max_val, bins)

        # Calculate probability distributions
        p, _ = np.histogram(baseline, bins=bins_range, density=True)
        q, _ = np.histogram(current, bins=bins_range, density=True)

        # Normalize to ensure they sum to 1
        p = p / p.sum() if p.sum() > 0 else p + 1e-10
        q = q / q.sum() if q.sum() > 0 else q + 1e-10

        # Calculate JS Divergence
        m = 0.5 * (p + q)
        js_div = 0.5 * entropy(p, m) + 0.5 * entropy(q, m)

        return js_div

    # Calculate JS Divergence for each feature
    js_results = {}
    for column in baseline_data.columns:
        js_value = calculate_js_divergence(
            baseline_data[column].dropna(),
            current_data[column].dropna()
        )
        js_results[column] = js_value

    # Aggregate
    max_js = max(js_results.values())
    avg_js = np.mean(list(js_results.values()))

    # Count features above threshold
    drifted_count = sum(1 for v in js_results.values() if v > js_threshold)

    return {
        'value': float(max_js),
        'metadata': {
            'average_js_divergence': float(avg_js),
            'feature_js': js_results,
            'drifted_features': drifted_count,
            'baseline_samples': len(baseline_data),
            'current_samples': len(current_data)
        },
        'status': 'critical' if max_js > js_threshold * 2
                  else 'warning' if max_js > js_threshold
                  else 'ok'
    }""", language="python")

                elif metric_type == "Custom Python Function":
                    custom_function = st.text_area(
                        "Python Function",
                        value="""def compute_metric(baseline_data, current_data):
    \"\"\"
    Compute a custom metric comparing baseline and current data.

    Args:
        baseline_data: pandas DataFrame with baseline predictions
            - Columns: feature names from model schema
            - Shape: (n_baseline_samples, n_features)
        current_data: pandas DataFrame with current predictions
            - Columns: same feature names as baseline_data
            - Shape: (n_current_samples, n_features)

    Returns:
        dict with:
            - 'value': float - The computed metric value
            - 'metadata': dict - Optional additional information
            - 'status': str - Optional status ('ok', 'warning', 'critical')

    Available libraries: numpy as np, pandas as pd, scipy.stats
    \"\"\"
    import numpy as np
    from scipy import stats

    # Example: Compute mean absolute difference in predictions
    baseline_mean = baseline_data.mean()
    current_mean = current_data.mean()
    metric_value = np.abs(baseline_mean - current_mean).mean()

    # Determine status based on threshold
    if metric_value < 0.05:
        status = 'ok'
    elif metric_value < 0.15:
        status = 'warning'
    else:
        status = 'critical'

    return {
        'value': float(metric_value),
        'metadata': {
            'baseline_samples': len(baseline_data),
            'current_samples': len(current_data),
            'baseline_mean': float(baseline_mean.mean()),
            'current_mean': float(current_mean.mean())
        },
        'status': status
    }""",
                        height=400,
                        help="Define a custom Python function to compute your metric",
                        key="custom_function_code"
                    )

                st.markdown("---")

                # Action Section
                st.markdown("#### 4️⃣ Compute & Save")

                col1, col2, col3 = st.columns([2, 2, 1])

                with col1:
                    # Generate metric name with underscores (no spaces allowed)
                    default_metric_name = metric_type.replace(" ", "_").replace("(", "").replace(")", "") + "_drift"
                    metric_name_input = st.text_input(
                        "Metric Name",
                        value=default_metric_name,
                        help="Name for this metric (alphanumeric, dashes, underscores, periods, tildes only)"
                    )
                    # Sanitize metric name: replace invalid characters with underscores
                    import re
                    metric_name = re.sub(r'[^a-zA-Z0-9\-_.~]', '_', metric_name_input)
                    if metric_name != metric_name_input:
                        st.caption(f"⚠️ Sanitized to: `{metric_name}`")

                with col2:
                    add_tags = st.checkbox("Add tags", value=False)
                    if add_tags:
                        tags_input = st.text_input(
                            "Tags (comma-separated)",
                            placeholder="experiment:v1, threshold:0.05",
                            help="Optional tags for organizing metrics"
                        )

                with col3:
                    st.markdown("") # Spacer
                    st.markdown("") # Spacer
                    compute_button = st.button(
                        "🚀 Compute Metric",
                        type="primary",
                        use_container_width=True
                    )

                # Results Section (shown after compute)
                if compute_button:
                    st.markdown("---")
                    st.markdown("### 📊 Results")

                    # Check if training set is confirmed when available
                    if TRAINING_SETS_AVAILABLE and selected_training_set and not training_set_confirmed:
                        st.error("⚠️ Please confirm the training set selection above before computing metrics.")
                        st.stop()

                    # Check if Domino SDK is available
                    if not DOMINO_AVAILABLE:
                        st.error("⚠️ Domino SDK not available. Cannot log custom metrics.")
                        st.info("Install with: `pip install domino`")
                        st.stop()

                    try:
                        with st.spinner("Loading data..."):
                            # Get the workbench model version ID for the selected model
                            selected_model = next((m for m in models if m.get('id') == selected_model_id), None)
                            workbench_model_version_id = None
                            if selected_model:
                                source_details = selected_model.get('sourceDetails', {})
                                workbench_model_version_id = source_details.get('workbenchModelVersionId')
                            
                            # Load baseline data (training set or fallback)
                            if TRAINING_SETS_AVAILABLE and selected_training_set and training_set_confirmed:
                                st.info("📊 Loading baseline data from training set")
                                baseline_data = load_training_set_data(selected_training_set)
                            else:
                                st.info("📊 Loading baseline data from prediction history (fallback)")
                                baseline_data = load_prediction_data(
                                    datetime.combine(baseline_start, datetime.min.time()),
                                    datetime.combine(baseline_end, datetime.max.time()),
                                    model_id=workbench_model_version_id or selected_model_id
                                )
                            
                            # Load current data (always from predictions)
                            current_data = load_prediction_data(
                                datetime.combine(current_start, datetime.min.time()),
                                datetime.combine(current_end, datetime.max.time()),
                                model_id=workbench_model_version_id or selected_model_id
                            )

                            if baseline_data.empty or current_data.empty:
                                st.error("❌ Failed to load prediction data. Check data paths.")
                                st.stop()

                            # Select only numeric features for drift analysis
                            numeric_cols = baseline_data.select_dtypes(include=[np.number]).columns
                            baseline_data = baseline_data[numeric_cols]
                            current_data = current_data[numeric_cols]

                            st.success(f"✅ Loaded {len(baseline_data)} baseline and {len(current_data)} current samples")
                            
                            # Data Source Comparison Summary
                            with st.expander("📋 Data Source Summary", expanded=True):
                                col_summary1, col_summary2 = st.columns(2)
                                
                                with col_summary1:
                                    st.markdown("### 📚 Baseline Data Source")
                                    if TRAINING_SETS_AVAILABLE and selected_training_set and training_set_confirmed:
                                        st.markdown("**Source Type:** 🎯 Training Set (API)")
                                        st.markdown(f"**Training Set:** {selected_training_set.get('name', 'Unknown')}")
                                        st.markdown(f"**Samples:** {len(baseline_data):,}")
                                        st.markdown(f"**Features:** {len(baseline_data.columns)}")
                                        st.markdown("**Description:** Actual data used to train the model")
                                    else:
                                        st.markdown("**Source Type:** 📂 Prediction History (Fallback)")
                                        st.markdown(f"**Date Range:** {baseline_start} to {baseline_end}")
                                        st.markdown(f"**Samples:** {len(baseline_data):,}")
                                        st.markdown(f"**Features:** {len(baseline_data.columns)}")
                                        st.markdown("**Description:** Historical prediction data as proxy")
                                
                                with col_summary2:
                                    st.markdown("### 🔄 Current Data Source")
                                    st.markdown("**Source Type:** 📈 Prediction Data")
                                    st.markdown(f"**Date Range:** {current_start} to {current_end}")
                                    st.markdown(f"**Samples:** {len(current_data):,}")
                                    st.markdown(f"**Features:** {len(current_data.columns)}")
                                    if selected_model_id:
                                        # Find the workbench model version ID for this monitored model
                                        selected_model = next((m for m in models if m.get('id') == selected_model_id), None)
                                        if selected_model:
                                            source_details = selected_model.get('sourceDetails', {})
                                            workbench_version_id = source_details.get('workbenchModelVersionId')
                                            if workbench_version_id:
                                                prediction_dir = f"/mnt/data/prediction_data/{workbench_version_id}"
                                                if Path(prediction_dir).exists():
                                                    st.markdown(f"**Description:** Actual model predictions from `/mnt/data/prediction_data/{workbench_version_id}`")
                                                else:
                                                    st.markdown("**Description:** Fallback data (ground truth or training proxy)")
                                            else:
                                                st.markdown("**Description:** Fallback data (no workbench version ID)")
                                        else:
                                            st.markdown("**Description:** Model prediction data or fallback")
                                    else:
                                        st.markdown("**Description:** Model prediction data or fallback")
                                    
                                # Show tip when fallback mode is used
                                if not (TRAINING_SETS_AVAILABLE and selected_training_set and training_set_confirmed):
                                    st.info("💡 **Tip:** Select a training set above for more accurate baseline comparison against actual training data.")

                        with st.spinner("Computing metric..."):
                            # Execute the appropriate metric function
                            result = None

                            if metric_type == "Kolmogorov-Smirnov Test":
                                # Import required libraries
                                from scipy import stats
                                import numpy as np

                                results = []
                                for column in baseline_data.columns:
                                    ks_stat, p_value = stats.ks_2samp(
                                        baseline_data[column].dropna(),
                                        current_data[column].dropna()
                                    )
                                    results.append({
                                        'feature': column,
                                        'ks_statistic': ks_stat,
                                        'p_value': p_value,
                                        'drifted': ks_stat > ks_threshold or p_value < significance_level
                                    })

                                drifted_features = [r for r in results if r['drifted']]
                                max_ks = max([r['ks_statistic'] for r in results])

                                result = {
                                    'value': float(max_ks),
                                    'metadata': {
                                        'total_features': len(results),
                                        'drifted_features': len(drifted_features),
                                        'feature_results': results,
                                        'baseline_samples': len(baseline_data),
                                        'current_samples': len(current_data)
                                    },
                                    'status': 'critical' if len(drifted_features) > len(results) * 0.3
                                              else 'warning' if len(drifted_features) > 0
                                              else 'ok'
                                }

                            elif metric_type == "Population Stability Index (PSI)":
                                import numpy as np

                                def calculate_psi(baseline, current, bins=10):
                                    breakpoints = np.linspace(baseline.min(), baseline.max(), bins + 1)
                                    breakpoints[0] = -np.inf
                                    breakpoints[-1] = np.inf

                                    baseline_binned = pd.cut(baseline, bins=breakpoints)
                                    current_binned = pd.cut(current, bins=breakpoints)

                                    baseline_counts = baseline_binned.value_counts().sort_index()
                                    current_counts = current_binned.value_counts().sort_index()

                                    baseline_props = baseline_counts / len(baseline)
                                    current_props = current_counts / len(current)

                                    baseline_props = baseline_props.replace(0, 0.0001)
                                    current_props = current_props.replace(0, 0.0001)

                                    psi = np.sum((current_props - baseline_props) *
                                                 np.log(current_props / baseline_props))
                                    return psi

                                psi_results = {}
                                for column in baseline_data.columns:
                                    psi_value = calculate_psi(
                                        baseline_data[column].dropna(),
                                        current_data[column].dropna(),
                                        bins=num_bins
                                    )
                                    psi_results[column] = float(psi_value)

                                max_psi = max(psi_results.values())
                                avg_psi = np.mean(list(psi_results.values()))

                                result = {
                                    'value': float(max_psi),
                                    'metadata': {
                                        'average_psi': float(avg_psi),
                                        'feature_psi': psi_results,
                                        'baseline_samples': len(baseline_data),
                                        'current_samples': len(current_data)
                                    },
                                    'status': 'critical' if max_psi > psi_critical
                                              else 'warning' if max_psi > psi_threshold
                                              else 'ok'
                                }

                            elif metric_type == "Jensen-Shannon Divergence":
                                import numpy as np
                                from scipy.stats import entropy

                                def calculate_js_divergence(baseline, current, bins=50):
                                    min_val = min(baseline.min(), current.min())
                                    max_val = max(baseline.max(), current.max())
                                    bins_range = np.linspace(min_val, max_val, bins)

                                    p, _ = np.histogram(baseline, bins=bins_range, density=True)
                                    q, _ = np.histogram(current, bins=bins_range, density=True)

                                    p = p / p.sum() if p.sum() > 0 else p + 1e-10
                                    q = q / q.sum() if q.sum() > 0 else q + 1e-10

                                    m = 0.5 * (p + q)
                                    js_div = 0.5 * entropy(p, m) + 0.5 * entropy(q, m)
                                    return js_div

                                js_results = {}
                                for column in baseline_data.columns:
                                    js_value = calculate_js_divergence(
                                        baseline_data[column].dropna(),
                                        current_data[column].dropna()
                                    )
                                    js_results[column] = float(js_value)

                                max_js = max(js_results.values())
                                avg_js = np.mean(list(js_results.values()))
                                drifted_count = sum(1 for v in js_results.values() if v > js_threshold)

                                result = {
                                    'value': float(max_js),
                                    'metadata': {
                                        'average_js_divergence': float(avg_js),
                                        'feature_js': js_results,
                                        'drifted_features': drifted_count,
                                        'baseline_samples': len(baseline_data),
                                        'current_samples': len(current_data)
                                    },
                                    'status': 'critical' if max_js > js_threshold * 2
                                              else 'warning' if max_js > js_threshold
                                              else 'ok'
                                }

                            elif metric_type == "Custom Python Function":
                                # Execute custom function
                                try:
                                    # Create a namespace with required imports
                                    namespace = {
                                        'pd': pd,
                                        'np': np,
                                        'baseline_data': baseline_data,
                                        'current_data': current_data
                                    }

                                    # Execute the custom function code
                                    exec(custom_function, namespace)
                                    compute_metric = namespace['compute_metric']

                                    # Call the function
                                    result = compute_metric(baseline_data, current_data)

                                except Exception as e:
                                    st.error(f"❌ Error executing custom function: {e}")
                                    st.code(str(e))
                                    st.stop()

                            if result is None:
                                st.error("❌ Metric computation failed")
                                st.stop()

                        # Log metric to Domino
                        with st.spinner("Logging metric to Domino..."):
                            try:
                                # Initialize Domino client (requires project in format "owner/project-name")
                                d = domino.Domino(
                                    config.DOMINO_PROJECT,
                                    api_key=config.API_KEY,
                                    host=config.DOMINO_SDK_HOST
                                )
                                metrics_client = d.custom_metrics_client()

                                # Prepare timestamp (current time in ISO format)
                                timestamp = datetime.utcnow().isoformat() + 'Z'

                                # Prepare tags
                                tags = {
                                    'metric_type': metric_type,
                                    'baseline_start': str(baseline_start),
                                    'baseline_end': str(baseline_end),
                                    'current_start': str(current_start),
                                    'current_end': str(current_end)
                                }

                                if add_tags and 'tags_input' in locals():
                                    # Parse comma-separated tags
                                    for tag_pair in tags_input.split(','):
                                        if ':' in tag_pair:
                                            key, value = tag_pair.split(':', 1)
                                            tags[key.strip()] = value.strip()

                                # Log the metric
                                metrics_client.log_metric(
                                    selected_model_id,
                                    metric_name,
                                    result['value'],
                                    timestamp,
                                    tags
                                )

                                st.success(f"✅ Metric '{metric_name}' logged successfully!")

                            except Exception as e:
                                st.warning(f"⚠️ Failed to log metric to Domino: {e}")
                                st.info("Displaying results anyway...")

                        # Display results
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Metric Value", f"{result['value']:.4f}", help="Computed metric value")
                        with col2:
                            status_display = {
                                'ok': '✅ OK',
                                'warning': '⚠️ Warning',
                                'critical': '🚨 Critical'
                            }.get(result.get('status', 'ok'), '❓ Unknown')
                            st.metric("Status", status_display)
                        with col3:
                            total_samples = result['metadata'].get('baseline_samples', 0) + result['metadata'].get('current_samples', 0)
                            st.metric("Total Samples", f"{total_samples:,}", help="Baseline + Current samples")

                        # Details in expander
                        with st.expander("📋 View Detailed Results", expanded=True):
                            st.markdown("**Metric Details:**")
                            st.json(result['metadata'])

                    except Exception as e:
                        st.error(f"❌ Error computing metric: {e}")
                        st.exception(e)

            with tab2:
                st.markdown("### Metric History")

                # Filters
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    history_model_name = st.selectbox(
                        "Filter by Model",
                        options=list(model_options.keys()),
                        key="history_model"
                    )
                    history_model_id = model_options[history_model_name]
                with col2:
                    # Provide dropdown with common metric names from the app
                    metric_options = [
                        "Kolmogorov-Smirnov_Test_drift",
                        "Population_Stability_Index_PSI_drift",
                        "Jensen-Shannon_Divergence_drift",
                        "Custom_Python_Function_drift",
                        "Other (enter manually)"
                    ]

                    selected_metric_option = st.selectbox(
                        "Select Metric *",
                        options=metric_options,
                        help="Choose a metric generated by this app, or select 'Other' to enter manually",
                        key="metric_dropdown"
                    )

                    # If "Other" is selected, show text input
                    if selected_metric_option == "Other (enter manually)":
                        history_metric_name = st.text_input(
                            "Enter Metric Name",
                            value="",
                            placeholder="e.g., my_custom_metric",
                            help="Enter the exact metric name (case-sensitive, use underscores)",
                            key="history_metric_name_input"
                        )
                    else:
                        history_metric_name = selected_metric_option
                with col3:
                    history_days_option = st.selectbox(
                        "Time Range",
                        options=["Last 7 days", "Last 30 days", "Last 90 days", "Last 365 days"],
                        index=1,
                        key="history_days"
                    )
                    # Convert to days
                    days_map = {
                        "Last 7 days": 7,
                        "Last 30 days": 30,
                        "Last 90 days": 90,
                        "Last 365 days": 365
                    }
                    history_days = days_map[history_days_option]
                with col4:
                    st.markdown("") # Spacer
                    st.markdown("") # Spacer
                    fetch_history_button = st.button("🔍 Fetch History", type="primary", key="fetch_history")

                st.markdown("---")

                # Fetch and display metrics if button clicked
                if fetch_history_button:
                    # Validate metric name is provided
                    if not history_metric_name or (selected_metric_option == "Other (enter manually)" and not history_metric_name.strip()):
                        st.error("⚠️ Please select a metric or enter a custom metric name.")
                        st.info("💡 Tip: Select from the dropdown or choose 'Other' and enter the exact name you used when computing the metric")
                        st.stop()

                    if not DOMINO_AVAILABLE:
                        st.error("⚠️ Domino SDK not available. Cannot fetch custom metrics.")
                        st.stop()

                    try:
                        with st.spinner("Fetching metric history from Domino..."):
                            # Initialize Domino client
                            d = domino.Domino(
                                config.DOMINO_PROJECT,
                                api_key=config.API_KEY,
                                host=config.DOMINO_SDK_HOST
                            )
                            metrics_client = d.custom_metrics_client()

                            # Calculate date range
                            import rfc3339
                            end_date = datetime.utcnow()
                            start_date = end_date - timedelta(days=history_days)

                            start_date_str = rfc3339.rfc3339(start_date)
                            end_date_str = rfc3339.rfc3339(end_date)

                            # Fetch metrics (workaround for SDK bug - call API directly)
                            # Use SDK's internal routing to build URL
                            url = metrics_client._routes.read_metrics(history_model_id, history_metric_name)
                            params = {
                                "startingReferenceTimestampInclusive": start_date_str,
                                "endingReferenceTimestampInclusive": end_date_str,
                            }

                            # Call API directly and get raw JSON (bypasses SDK's broken conversion)
                            response = metrics_client._parent.request_manager.get(url, params=params)
                            data = response.json()

                            # Extract metric values from raw JSON
                            metric_values = data.get('metricValues', [])

                            if metric_values:
                                # Convert to DataFrame
                                df_history = pd.DataFrame(metric_values)

                                # Rename referenceTimestamp to timestamp for consistency
                                if 'referenceTimestamp' in df_history.columns:
                                    df_history.rename(columns={'referenceTimestamp': 'timestamp'}, inplace=True)

                                # Parse timestamps
                                df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])

                                # Add metric name column
                                df_history['metric'] = history_metric_name

                                # Extract tags if present
                                if 'tags' in df_history.columns:
                                    # Tags is a list of dicts with 'key' and 'value'
                                    def extract_tags(tags_list):
                                        if not tags_list:
                                            return {}
                                        result = {}
                                        for tag in tags_list:
                                            if isinstance(tag, dict):
                                                result[tag.get('key')] = tag.get('value')
                                        return result

                                    df_history['tags_dict'] = df_history['tags'].apply(extract_tags)

                                    # Extract metric_type from tags
                                    df_history['metric_type'] = df_history['tags_dict'].apply(
                                        lambda x: x.get('metric_type', 'Unknown')
                                    )

                                # Convert Decimal to float for display
                                if 'value' in df_history.columns:
                                    df_history['value'] = df_history['value'].astype(float)

                                # Sort by timestamp
                                df_history = df_history.sort_values('timestamp', ascending=False)

                                st.success(f"✅ Found {len(df_history)} metric records")

                                # Display recent computations table
                                st.markdown("#### Recent Computations")
                                display_df = df_history[['timestamp', 'metric', 'value', 'metric_type']].copy() if 'metric_type' in df_history.columns else df_history[['timestamp', 'metric', 'value']].copy()
                                display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                                display_df['value'] = display_df['value'].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
                                st.dataframe(display_df, use_container_width=True)

                                # Plot trend chart if multiple records
                                if len(df_history) > 1:
                                    st.markdown("#### Metric Trends Over Time")

                                    # Single metric - simple line chart
                                    fig = px.line(
                                        df_history,
                                        x='timestamp',
                                        y='value',
                                        title=f'Metric Trend Over Time',
                                        template='plotly_dark',
                                        markers=True
                                    )
                                    fig.update_layout(
                                        xaxis_title='Timestamp',
                                        yaxis_title='Metric Value',
                                        height=400
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("📊 Need at least 2 data points to show trend chart")
                            else:
                                st.info(f"📭 No metrics found for '{history_metric_name}' in the selected time range.\n\n**Suggestions:**\n- Verify the metric name is spelled correctly (case-sensitive)\n- Try a longer time range\n- Compute the metric first if you haven't already\n- Check that you're using the correct model")

                    except ImportError as e:
                        st.error(f"❌ Missing required library: {e}")
                        st.info("Install with: `pip install rfc3339`")
                    except Exception as e:
                        st.error(f"❌ Failed to fetch metric history: {e}")
                        st.exception(e)
                else:
                    st.info("👆 Select a model and click 'Fetch History' to view custom metrics")

    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.exception(e)

# ==================== Footer ====================

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="text-align: center; color: #97A3B7; font-size: 0.85rem;">'
    'Model Monitoring API v2'
    '</div>',
    unsafe_allow_html=True
)
