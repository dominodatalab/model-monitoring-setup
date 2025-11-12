#Here's an example Python function to test in the app custom metric:

import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance

def compute_metric(baseline_data, current_data):
  """
  Wasserstein Distance (Earth Mover's Distance) for drift detection.
  
  Measures the minimum cost to transform one distribution into another.
  Values closer to 0 indicate less drift, higher values indicate more drift.
  
  Returns:
      float: Wasserstein distance between distributions
  """
  # Handle both pandas Series and numpy arrays
  if hasattr(baseline_data, 'values'):
      baseline_values = baseline_data.values
  else:
      baseline_values = np.array(baseline_data)

  if hasattr(current_data, 'values'):
      current_values = current_data.values
  else:
      current_values = np.array(current_data)

  # Remove any NaN values
  baseline_clean = baseline_values[~np.isnan(baseline_values)]
  current_clean = current_values[~np.isnan(current_values)]

  if len(baseline_clean) == 0 or len(current_clean) == 0:
      return {
          'value': 1.0,
          'metadata': {
              'baseline_samples': len(baseline_clean),
              'current_samples': len(current_clean),
              'metric_type': 'Wasserstein Distance',
              'error': 'No valid data available'
          },
          'status': 'critical'
      }

  # Calculate Wasserstein distance
  distance = wasserstein_distance(baseline_clean, current_clean)

  # Determine status based on distance thresholds
  if distance < 0.1:
      status = 'ok'
  elif distance < 0.3:
      status = 'warning'
  else:
      status = 'critical'

  # Return dictionary format expected by the app
  return {
      'value': float(distance),
      'metadata': {
          'baseline_samples': len(baseline_clean),
          'current_samples': len(current_clean),
          'metric_type': 'Wasserstein Distance'
      },
      'status': status
  }