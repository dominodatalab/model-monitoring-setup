"""
Configuration template for Model Monitoring Dashboard

Copy this file to config.py and fill in the URL
"""

# Domino API Configuration
#Fill in URL
DOMINO_API_HOST = "https://your-domino-url" #do not put in the trailing /

#--------------------------------------------------------------------------------------------------------
API_KEY = os.environ.get('DOMINO_USER_API_KEY', '') #would need to be modified if using outside of Domino

# Domino Project Configuration (for custom metrics SDK) - can fill in defauls if desired
DOMINO_PROJECT_OWNER = os.environ.get('DOMINO_PROJECT_OWNER', '')
DOMINO_PROJECT_NAME = os.environ.get('DOMINO_PROJECT_NAME', '')
DOMINO_PROJECT = f"{DOMINO_PROJECT_OWNER}/{DOMINO_PROJECT_NAME}"

# Domino SDK requires external URL (not internal nucleus URL)
DOMINO_SDK_HOST = API_HOST
