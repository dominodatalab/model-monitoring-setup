"""
Model Script for Domino Model API Deployment

This script loads the trained Random Forest model and provides prediction
functionality with integrated monitoring capture for Domino Model Monitor.

Deploy this as a Domino Model API endpoint.
"""

import joblib
import uuid
import os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

# Import DataCaptureClient for monitoring
try:
    from domino_data_capture.data_capture_client import DataCaptureClient
    MONITORING_ENABLED = True
except ImportError:
    MONITORING_ENABLED = False
    print("⚠️  Model Monitoring disabled - domino_data_capture not available")

# Find the model file in the correct location
def find_model_path():
    """Find the model.pkl file in different Domino environments"""
    # For Domino endpoints, use DOMINO_WORKING_DIR
    if 'DOMINO_WORKING_DIR' in os.environ:
        model_path = Path(os.environ['DOMINO_WORKING_DIR']) / 'example' / 'model.pkl'
        if model_path.exists():
            return str(model_path)
    
    # For workspaces/runs, try relative paths
    possible_paths = [
        'model.pkl',  # Current directory
        '../model.pkl',  # Parent directory
        'example/model.pkl',  # Example subdirectory
        '/mnt/artifacts/model.pkl',  # Artifacts directory
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            return path
    
    raise FileNotFoundError("model.pkl not found in any expected location")

# Load the trained model at module level
model_path = find_model_path()
print(f"Loading model from: {model_path}")
model = joblib.load(model_path)

# Define feature names (must match training data order)
feature_names = [
    '1_feature', '2_feature', '3_feature', '4_feature', '5_feature',
    '6_feature', '7_feature', '8_feature', '9_feature', '10_feature'
]

# Define prediction names
predict_names = ['predicted_class', 'confidence_score']

# Initialize DataCaptureClient once at module level
if MONITORING_ENABLED:
    data_capture_client = DataCaptureClient(feature_names, predict_names)
else:
    data_capture_client = None


def predict(**kwargs):
    """
    Main prediction function with monitoring capture.
    
    Args:
        **kwargs: Feature values passed as keyword arguments, e.g.:
            1_feature=0.5, 2_feature=-0.2, etc.
            OR
            input_data dictionary with all features
    
    Returns:
        Dictionary with prediction results:
            {
                "predicted_class": "class_1",
                "confidence_score": 0.85,
                "probabilities": [0.1, 0.85, 0.05, 0.0],
                "event_id": "uuid-string",
                "timestamp": "iso-timestamp"
            }
    """
    # Handle different input formats
    if len(kwargs) == 1 and 'input_data' in kwargs:
        # Called with input_data dictionary (for testing)
        input_data = kwargs['input_data']
    else:
        # Called with features as keyword arguments (Domino API format)
        input_data = kwargs
    
    # Extract features in the correct order
    feature_values = []
    for feature_name in feature_names:
        if feature_name not in input_data:
            raise ValueError(f"Missing required feature: {feature_name}")
        feature_values.append(input_data[feature_name])
    
    # Convert to numpy array for prediction
    import pandas as pd
    feature_df = pd.DataFrame([feature_values], columns=feature_names)
    
    # Make prediction
    predicted_class = model.predict(feature_df)[0]
    class_probabilities = model.predict_proba(feature_df)[0].tolist()
    
    # Get confidence score (max probability)
    confidence_score = max(class_probabilities)
    
    # Prediction values to capture for monitoring
    predict_values = [predicted_class, confidence_score]
    
    # Generate unique event ID and timestamp
    event_id = str(uuid.uuid4())
    event_time = datetime.now(timezone.utc).isoformat()
    
    # Capture prediction for monitoring
    if MONITORING_ENABLED and data_capture_client is not None:
        try:
            data_capture_client.capturePrediction(
                feature_values,
                predict_values,
                event_id=event_id,
                timestamp=event_time,
                prediction_probability=class_probabilities  # Enable AUC/log loss metrics
            )
        except Exception as e:
            # Don't fail prediction if monitoring fails
            print(f"⚠️  Monitoring capture failed: {e}")
    
    # Return prediction response
    return {
        "predicted_class": predicted_class,
        "confidence_score": confidence_score,
        "probabilities": class_probabilities,
        "event_id": event_id,
        "timestamp": event_time
    }


def health_check():
    """
    Health check endpoint for Domino Model API.
    """
    return {"status": "healthy", "model_loaded": model is not None}


# Test function for local development
if __name__ == "__main__":
    # Test with sample data
    test_input = {
        '1_feature': 0.5,
        '2_feature': -0.2,
        '3_feature': 1.1,
        '4_feature': 0.0,
        '5_feature': -0.8,
        '6_feature': 0.3,
        '7_feature': 1.2,
        '8_feature': -0.1,
        '9_feature': 0.7,
        '10_feature': -0.5
    }
    
    result = predict(input_data=test_input)
    print("Test prediction:")
    print(f"  Class: {result['predicted_class']}")
    print(f"  Confidence: {result['confidence_score']:.3f}")
    print(f"  Probabilities: {[round(p, 3) for p in result['probabilities']]}")
    print(f"  Event ID: {result['event_id']}")