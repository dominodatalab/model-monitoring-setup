#!/usr/bin/env python3
"""
Configuration Loader for Model Monitoring

Centralized configuration management for monitoring scripts.
"""

import os
import json
from pathlib import Path
from typing import Optional


class MonitoringConfig:
    """Configuration manager for model monitoring"""

    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration loader"""
        self.config_file = self._find_config_file(config_file)
        self.config = self._load_config()
        self._validate_config()

    def _find_config_file(self, config_file: Optional[str] = None) -> Path:
        """Find the configuration file"""
        if config_file:
            return Path(config_file)

        candidates = [
            Path('/mnt/artifacts/monitoring_config.json'),
            Path('monitoring_config.json'),
            Path('/mnt/monitoring_config.json')
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return candidates[0]

    def _load_config(self):
        """Load configuration from JSON"""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Config not found: {self.config_file}\n"
                f"Run: python setup_monitoring.py"
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {self.config_file}: {e}")

    def _validate_config(self):
        """Validate required fields"""
        required = ['domino', 'model_api', 'model_monitor', 'data_sources']
        missing = [s for s in required if s not in self.config]

        if missing:
            raise ValueError(f"Missing config sections: {missing}")

    @property
    def domino_base_url(self) -> str:
        return self.config['domino']['base_url'].rstrip('/')

    @property
    def domino_api_key(self) -> str:
        env_var = self.config['domino'].get('user_api_key_env', 'DOMINO_USER_API_KEY')
        api_key = os.environ.get(env_var)
        if not api_key:
            raise ValueError(
                f"API key not found in environment variable: {env_var}\n"
                f"This should be automatically available in Domino workspaces."
            )
        return api_key

    @property
    def model_api_url(self) -> str:
        return self.config['model_api']['endpoint_url']

    @property
    def model_api_token(self) -> str:
        return self.config['model_api']['token']

    @property
    def model_monitor_id(self) -> str:
        return self.config['model_monitor']['model_id']

    @property
    def ground_truth_datasource(self) -> str:
        return self.config['data_sources']['ground_truth']

    @property
    def test_data_path(self) -> str:
        return self.config.get('test_data', {}).get('path', '')

    @property
    def model_monitor_api_url(self) -> str:
        return f"{self.domino_base_url}/model-monitor/v2/api"


_config_instance = None


def get_config(config_file: Optional[str] = None):
    """Get global configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = MonitoringConfig(config_file)
    return _config_instance
