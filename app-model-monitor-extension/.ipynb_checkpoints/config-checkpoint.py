"""
Configuration for Model Monitoring Dashboard

This file loads configuration from environment variables or defaults.
"""

import os

# Domino API Configuration (for Model Monitoring API v2)
API_HOST = 'https://ews.domino-eval.com'  # External URL required for Model Monitoring v2
API_KEY = os.environ.get('DOMINO_USER_API_KEY', '')

# Domino Project Configuration (for custom metrics SDK)
DOMINO_PROJECT_OWNER = os.environ.get('DOMINO_PROJECT_OWNER', 'integration-test')
DOMINO_PROJECT_NAME = os.environ.get('DOMINO_PROJECT_NAME', 'Fraud-Detection-Workshop')
DOMINO_PROJECT = f"{DOMINO_PROJECT_OWNER}/{DOMINO_PROJECT_NAME}"

# Domino SDK requires external URL (not internal nucleus URL)
DOMINO_SDK_HOST = 'https://ews.domino-eval.com'
