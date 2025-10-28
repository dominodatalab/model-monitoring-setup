#!/usr/bin/env python3
"""
Model Monitoring API v2 Client

Handles all interactions with Domino Model Monitoring API v2.
Only shows models registered in the Model Monitor system.
"""

import os
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd


class ModelMonitoringClient:
    """Client for Domino Model Monitoring API v2."""

    def __init__(self, api_host: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the Model Monitoring client.

        Args:
            api_host: Domino API host (defaults to external URL)
            api_key: Domino API key (defaults to env var)
        """
        # Model Monitoring API requires external URL
        self.api_host = api_host or os.environ.get('DOMINO_API_HOST', 'https://ews.domino-eval.com')

        # If we got the internal URL, convert to external
        if 'nucleus-frontend' in self.api_host:
            self.api_host = 'https://ews.domino-eval.com'

        self.api_key = api_key or os.environ.get('DOMINO_USER_API_KEY', '')

        if not self.api_key:
            raise ValueError("API key not configured. Set DOMINO_USER_API_KEY or provide api_key parameter.")

        self.base_url = f"{self.api_host}/model-monitor/v2/api"
        self.headers = {
            "Content-Type": "application/json",
            "X-Domino-Api-Key": self.api_key
        }

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make an API request with error handling.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, PATCH)
            endpoint: API endpoint path (without base URL)
            **kwargs: Additional arguments for requests

        Returns:
            Response JSON data

        Raises:
            requests.HTTPError: On API errors
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self.headers,
                timeout=30,
                **kwargs
            )
            response.raise_for_status()

            # Handle empty responses
            if response.status_code == 204 or not response.content:
                return {}

            return response.json()

        except requests.exceptions.HTTPError as e:
            # Try to extract error message from response
            try:
                error_data = e.response.json()
                error_msg = error_data.get('message', str(error_data))
            except:
                error_msg = e.response.text[:500] if e.response.text else str(e)

            raise requests.HTTPError(
                f"API request failed: {e.response.status_code} - {error_msg}"
            ) from e

    # ==================== Model Management ====================

    def list_models(
        self,
        page_number: int = 0,
        page_size: int = 100,
        search_query: Optional[str] = None,
        model_type: Optional[str] = None,
        sort_on: str = "created_at",
        sort_order: int = -1,
        number_of_last_checks: int = 10
    ) -> Dict[str, Any]:
        """
        List all Model Monitoring models with pagination.

        Args:
            page_number: Page number (0-indexed)
            page_size: Number of models per page
            search_query: Optional search query
            model_type: Filter by model type (classification/regression)
            sort_on: Field to sort on (created_at, updated_at, etc.)
            sort_order: Sort order (1=asc, -1=desc)
            number_of_last_checks: Number of recent checks to include

        Returns:
            Dict with model dashboard items and pagination info
        """
        params = {
            "pageNumber": page_number,
            "pageSize": page_size,
            "numberOfLastChecksToFetch": number_of_last_checks,
            "sortOn": sort_on,
            "sortOrder": sort_order
        }

        if search_query:
            params["searchQuery"] = search_query

        if model_type:
            params["modelType"] = model_type

        return self._make_request("GET", "/models", params=params)

    def get_model(self, model_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.

        Args:
            model_id: Model ID

        Returns:
            Model details dictionary
        """
        params = {"model_id": model_id}
        return self._make_request("GET", "/model", params=params)

    # ==================== Drift Analysis ====================

    def get_drift_summary(
        self,
        model_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get drift summary for a model.

        Args:
            model_id: Model ID
            start_date: Start date (ISO format)
            end_date: End date (ISO format)

        Returns:
            Drift summary data
        """
        # Default to last 30 days if not specified
        if not end_date:
            end_date = datetime.utcnow().isoformat() + 'Z'
        if not start_date:
            start_date = (datetime.utcnow() - timedelta(days=30)).isoformat() + 'Z'

        params = {
            "startDate": start_date,
            "endDate": end_date
        }

        return self._make_request(
            "GET",
            f"/model/{model_id}/drift-summary",
            params=params
        )

    def get_drift_trend(
        self,
        model_id: str,
        column_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        algorithm: str = "psi"
    ) -> Dict[str, Any]:
        """
        Get drift trend data for a specific feature column.

        Args:
            model_id: Model ID
            column_id: Column/feature ID
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            algorithm: Drift algorithm (psi, ks, etc.)

        Returns:
            Drift trend data
        """
        # Default to last 30 days if not specified
        if not end_date:
            end_date = datetime.utcnow().isoformat() + 'Z'
        if not start_date:
            start_date = (datetime.utcnow() - timedelta(days=30)).isoformat() + 'Z'

        params = {
            "startDate": start_date,
            "endDate": end_date,
            "algorithm": algorithm
        }

        return self._make_request(
            "GET",
            f"/model/{model_id}/{column_id}/drift-trend",
            params=params
        )

    # ==================== Ground Truth & Traffic ====================

    def get_ground_truth_traffic(
        self,
        model_id: str,
        start_timestamp: Optional[int] = None,
        end_timestamp: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get ground truth traffic data for a model.

        Args:
            model_id: Model ID
            start_timestamp: Start timestamp (Unix timestamp in seconds)
            end_timestamp: End timestamp (Unix timestamp in seconds)

        Returns:
            Ground truth traffic data
        """
        # Default to last 30 days if not specified
        if not end_timestamp:
            end_timestamp = int(datetime.utcnow().timestamp())
        if not start_timestamp:
            start_timestamp = int((datetime.utcnow() - timedelta(days=30)).timestamp())

        params = {
            "startTimestamp": start_timestamp,
            "endTimestamp": end_timestamp
        }

        return self._make_request(
            "GET",
            f"/model/{model_id}/traffic/ground-truth",
            params=params
        )

    def get_prediction_traffic(
        self,
        model_id: str,
        start_timestamp: Optional[int] = None,
        end_timestamp: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get prediction traffic data for a model.

        Args:
            model_id: Model ID (or list of IDs)
            start_timestamp: Start timestamp (Unix timestamp in seconds)
            end_timestamp: End timestamp (Unix timestamp in seconds)

        Returns:
            Prediction traffic data (array with traffic per model)
        """
        # Default to last 30 days if not specified
        if not end_timestamp:
            end_timestamp = int(datetime.utcnow().timestamp())
        if not start_timestamp:
            start_timestamp = int((datetime.utcnow() - timedelta(days=30)).timestamp())

        params = {
            "model_ids": model_id,
            "start_date": start_timestamp,  # Unix timestamp in seconds
            "end_date": end_timestamp  # Unix timestamp in seconds
        }

        return self._make_request(
            "GET",
            "/model/get-prediction-traffic",
            params=params
        )

    # ==================== Model Quality ====================

    def get_model_quality(
        self,
        model_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get model quality metrics.

        Args:
            model_id: Model ID
            start_date: Start date (ISO format)
            end_date: End date (ISO format)

        Returns:
            Model quality data
        """
        # Default to last 30 days if not specified
        if not end_date:
            end_date = datetime.utcnow().isoformat() + 'Z'
        if not start_date:
            start_date = (datetime.utcnow() - timedelta(days=30)).isoformat() + 'Z'

        params = {
            "startDate": start_date,
            "endDate": end_date
        }

        return self._make_request(
            "GET",
            f"/model/{model_id}/model-quality",
            params=params
        )

    # ==================== Helper Methods ====================

    def get_models_summary(self) -> pd.DataFrame:
        """
        Get a summary of all Model Monitoring models as a DataFrame.

        Returns:
            DataFrame with model summary information
        """
        response = self.list_models(page_size=100)
        models = response.get('modelDashboardItems', [])

        if not models:
            return pd.DataFrame()

        # Extract key fields for summary
        summary_data = []
        for model in models:
            # Get latest drift check info
            drift_checks = model.get('dataDriftChecks', [])
            latest_drift = drift_checks[0] if drift_checks else {}

            # Get latest quality check info
            quality_checks = model.get('modelQualityChecks', [])
            latest_quality = quality_checks[0] if quality_checks else {}

            summary_data.append({
                'Model ID': model.get('id', ''),
                'Name': model.get('name', ''),
                'Version': model.get('version', ''),
                'Type': model.get('modelType', '').title(),
                'Status': model.get('modelStatus', ''),
                'Created': datetime.fromtimestamp(model.get('createdAt', 0)).isoformat() if model.get('createdAt') else '',
                'Drift Scheduled': 'Yes' if model.get('isDataDriftCheckScheduled') else 'No',
                'Quality Scheduled': 'Yes' if model.get('isModelQualityCheckScheduled') else 'No',
                'Variables Drifted': len(latest_drift.get('variablesDrifted', [])),
                'Last Check': datetime.fromtimestamp(latest_drift.get('checkedOn', 0)).isoformat() if latest_drift.get('checkedOn') else 'Never'
            })

        return pd.DataFrame(summary_data)
