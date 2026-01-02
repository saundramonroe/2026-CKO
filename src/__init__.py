"""
Fraud Detection System - Anaconda Core Package

This package provides a hybrid fraud detection system combining:
- Traditional ML (XGBoost/Random Forest)
- Large Language Model (Qwen 2.5 7B)

Modules:
    config: Configuration settings and constants
    data_utils: Data loading and feature engineering
    models: ML models and LLM integration
    api_client: Production API clients

Anaconda Value:
    - All dependencies managed by Anaconda Core
    - Reproducible environment via environment.yml
    - Seamless integration of ML + LLM
"""

__version__ = "1.0.0"
__author__ = "Anaconda Technical Enablement Team"
__email__ = "sales-enablement@anaconda.com"

# Import key components for easy access
from . import config
from . import data_utils
from . import models
from . import api_client

# Version info
VERSION_INFO = {
    'version': __version__,
    'author': __author__,
    'description': 'Hybrid fraud detection system',
    'anaconda_components': ['Core', 'Desktop', 'AI Catalyst']
}

def print_version():
    """Print package version information"""
    print(f"Fraud Detection System v{__version__}")
    print(f"Author: {__author__}")
    print(f"Anaconda Components: {', '.join(VERSION_INFO['anaconda_components'])}")

# Make commonly used items available at package level
__all__ = [
    'config',
    'data_utils', 
    'models',
    'api_client',
    'VERSION_INFO',
    'print_version'
]