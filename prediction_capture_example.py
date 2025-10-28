"""
Prediction Capture Example for Model Monitoring

Add this code to your model deployment script (e.g., predict.py) to enable
automatic prediction capture for Domino Model Monitor.

Requirements:
    pip install dominodatalab
"""

# Example deployment script with prediction capture
import uuid
from datetime import datetime, timezone

# Import DataCaptureClient
try:
    from domino_data_capture.data_capture_client import DataCaptureClient
    MONITORING_ENABLED = True
except ImportError:
    MONITORING_ENABLED = False
    print("⚠️  Model Monitoring disabled - domino_data_capture not available")


# Define your feature and prediction names
# NOTE: Feature names should match your training data columns
feature_names = [
    'feature_1',
    'feature_2',
    'feature_3'
    # Add all your feature names here
]

predict_names = ['predicted_class', 'confidence_score']

# Initialize DataCaptureClient once at module level
if MONITORING_ENABLED:
    data_capture_client = DataCaptureClient(feature_names, predict_names)
else:
    data_capture_client = None


def predict(input_data):
    """
    Main prediction function with monitoring capture.

    Args:
        input_data: Dictionary with feature values

    Returns:
        Dictionary with prediction results
    """
    # Your model loading and prediction logic here
    # model = load_your_model()
    # prediction = model.predict(input_data)

    # Example: Extract features from input
    feature_values = [
        input_data.get('feature_1'),
        input_data.get('feature_2'),
        input_data.get('feature_3')
        # Match the order of feature_names above
    ]

    # Example prediction results
    predicted_class = "class_A"  # Your model's prediction
    confidence_score = 0.95       # Your model's confidence

    # Prediction values to capture
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
                timestamp=event_time
            )
        except Exception as e:
            # Don't fail prediction if monitoring fails
            print(f"⚠️  Monitoring capture failed: {e}")

    # Return prediction response
    return {
        "predicted_class": predicted_class,
        "confidence_score": confidence_score,
        "event_id": event_id,
        "timestamp": event_time
    }


# ============================================================================
# MULTI-CLASS CLASSIFICATION WITH PROBABILITIES
# ============================================================================

def predict_multiclass_with_probabilities(input_data):
    """
    Example for multi-class classification with probability capture.
    Enables AUC-ROC and log loss metrics in Model Monitor.
    """
    # Extract features
    feature_values = [
        input_data.get('feature_1'),
        input_data.get('feature_2'),
        input_data.get('feature_3')
    ]

    # Your model prediction
    # predicted_class = model.predict(input_data)
    # class_probabilities = model.predict_proba(input_data)

    # Example results
    predicted_class = "class_A"
    confidence_score = 0.85

    # Class probabilities in consistent order (e.g., alphabetical)
    # Should be [prob_class_A, prob_class_B, prob_class_C]
    class_probabilities = [0.85, 0.10, 0.05]

    predict_values = [predicted_class, confidence_score]

    event_id = str(uuid.uuid4())
    event_time = datetime.now(timezone.utc).isoformat()

    # Capture with probabilities
    if MONITORING_ENABLED and data_capture_client is not None:
        try:
            data_capture_client.capturePrediction(
                feature_values,
                predict_values,
                event_id=event_id,
                timestamp=event_time,
                prediction_probability=class_probabilities  # Enable AUC/log loss
            )
        except Exception as e:
            print(f"⚠️  Monitoring capture failed: {e}")

    return {
        "predicted_class": predicted_class,
        "confidence_score": confidence_score,
        "probabilities": class_probabilities,
        "event_id": event_id,
        "timestamp": event_time
    }
