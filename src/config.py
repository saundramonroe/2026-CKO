# pylint: disable=simplifiable-if-expression
"""
Configuration for Fraud Detection System

This module centralizes all configuration settings for:
- API endpoints
- Model parameters
- Demo mode settings
- Sample merchants
- Performance thresholds

Anaconda Value: Single source of truth for reproducible experiments
"""

# ================================================================================
# API ENDPOINTS
# ================================================================================
# API Authentication Token for Anaconda Connect
API_TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3OTkzNDY3MjMsImtpZCI6IjMxIiwic2NvcGVzIjpbImNsb3VkOnJlYWQiLCJjbG91ZDp3cml0ZSIsInJlcG86cmVhZCIsInJlcG86d3JpdGUiXSwic3ViIjoiNzU5ZTgyNjUtZDlhZi00ZDRjLWI4NDQtYzZmMjk1Y2M2MzY4IiwidmVyIjoiYXBpOjEifQ.lKRLL1GYU3iKpATF4wXoT08a0L_3NT6OofL4P72xbOIG_9vMXWhsoi5RN40UsACn2wQZcJA_tMsf0JEfZiu7l2ACkO-Z28xSkeqCfPi8q15ZY6M0tYFMBrezRKdBa2Lhd1ULG6wGaJdBQZqnJnnhUa-oIV-cY43B0Y9x7EJoZKTPgDaPcmGNo0UJ-FM7INsiCxZUssUkY_9_92eh2O4m57_aJ190FNXnr11s6nbTynmnmKz3hK67WKjblpk-BVhKjnL9pCcyXkfWDVXviwNxlJwVCXHX7t-HbK1orXueLS-Vd91vZzgurhj6zoi9Ww_pJ0mi9_tmLyhHSXfFymO-Iw"

# Export it
__all__ = [
    'CONNECT_ENDPOINT', 
    'NAVIGATOR_ENDPOINT', 
    'API_TOKEN',
    'DEMO_MODE',
    'LEGITIMATE_MERCHANTS',
    'SUSPICIOUS_MERCHANTS'
]


# Anaconda Connect - Production deployed model
CONNECT_ENDPOINT = "https://demo.sb.anacondaconnect.com/api/ai/inference/serve/424c8a7d-5d52-4afb-bc60-81c23e837134"

# AI Navigator - Local LLM server
NAVIGATOR_ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"


# ================================================================================
# MODEL CONFIGURATION
# ================================================================================

# LLM Model
LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LLM_MAX_NEW_TOKENS = 5
LLM_TEMPERATURE = 0.1

# Hybrid Model Weights
MODEL_WEIGHTS = {
    'xgb': 0.6,  # XGBoost contribution
    'llm': 0.4   # LLM contribution
}

# XGBoost Parameters
XGB_N_ESTIMATORS = 100
XGB_MAX_DEPTH = 12
XGB_RANDOM_STATE = 42


# ================================================================================
# DEMO MODE CONFIGURATION
# ================================================================================

# Set to True for fast demos (5-10 min), False for full analysis (60+ min)
DEMO_MODE = True

# Training/Test Sizes
TRAIN_SAMPLE_SIZE = 50000 if DEMO_MODE else None  # Reduced from 227K
TEST_SAMPLE_SIZE = 10000 if DEMO_MODE else None   # Reduced from 57K

# LLM Analysis Limits (CRITICAL for speed)
LLM_ANALYSIS_LIMIT = 10 if DEMO_MODE else 100     # Max LLM calls in evaluation
USE_LLM_IN_EVALUATION = False if DEMO_MODE else True  # Skip LLM for fast eval

# Benchmark Configuration
BENCHMARK_SIZE = 20 if DEMO_MODE else 50

# Data Split
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ================================================================================
# MERCHANT SAMPLES
# ================================================================================

LEGITIMATE_MERCHANTS = [
    "AMAZON.COM MKTP US",
    "WALMART SUPERCENTER #1234",
    "STARBUCKS STORE #5678",
    "SHELL GAS STATION",
    "TARGET STORE #9012",
    "COSTCO WHOLESALE",
    "NETFLIX SUBSCRIPTION",
    "SPOTIFY PREMIUM",
    "UBER TRIP #12345",
    "APPLE.COM BILL",
    "WHOLE FOODS MARKET",
    "CVS PHARMACY #4567",
    "HOME DEPOT #8901",
    "TRADER JOES",
    "PANERA BREAD"
]

SUSPICIOUS_MERCHANTS = [
    "BITCOIN ATM UNKNOWN",
    "WIRE TRANSFER 7823",
    "ONLINE CASINO DEPOSIT",
    "CRYPTO EXCHANGE UNVERIFIED",
    "FOREIGN CODE 4456",
    "DUPLICATE CHARGE ALERT",
    "PAYPAL SUSPICIOUS",
    "UNKNOWN MERCHANT 9991",
    "OVERSEAS TRANSFER 3344",
    "UNVERIFIED PAYMENT",
    "DARK WEB MARKET",
    "ANONYMOUS TRANSFER",
    "HIGH RISK CASINO",
    "UNTRACEABLE EXCHANGE"
]


# ================================================================================
# PERFORMANCE THRESHOLDS
# ================================================================================

# Risk Score Thresholds
HIGH_RISK_THRESHOLD = 0.8   # Block transaction
MEDIUM_RISK_THRESHOLD = 0.5  # Review required
LOW_RISK_THRESHOLD = 0.3    # LLM trigger point

# SLA Requirements
MAX_LATENCY_MS = 100  # Production SLA
TARGET_ACCURACY = 0.85
TARGET_PRECISION = 0.85
TARGET_RECALL = 0.80


# ================================================================================
# FILE PATHS
# ================================================================================

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = Path(
    os.getenv("CREDITCARD_DATA_PATH", PROJECT_ROOT / "data" / "creditcard.csv")
)
MODEL_SAVE_PATH = PROJECT_ROOT / 'models' / 'hybrid_fraud_model.pkl'
RESULTS_PATH = PROJECT_ROOT / 'assets' / 'fraud_detection_results.png'


# ================================================================================
# ANACONDA VALUE METRICS
# ================================================================================

ANACONDA_VALUE = {
    'governance_time_reduction': '2-3 weeks → 20 minutes',
    'sbom_generation': '6-8 hours → 2 minutes',
    'deployment_time': 'hours → one-click',
    'package_approval': 'zero delays',
    'environment_reproducibility': '100%'
}


# ================================================================================
# HELPER FUNCTIONS
# ================================================================================

def get_demo_config():
    """Return current demo configuration as dict"""
    return {
        'demo_mode': DEMO_MODE,
        'train_samples': TRAIN_SAMPLE_SIZE or 'All',
        'test_samples': TEST_SAMPLE_SIZE or 'All',
        'llm_limit': LLM_ANALYSIS_LIMIT,
        'llm_in_eval': USE_LLM_IN_EVALUATION
    }

def print_config():
    """Print current configuration"""
    config = get_demo_config()
    print("=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print(f"Demo Mode: {config['demo_mode']}")
    print(f"Training Samples: {config['train_samples']}")
    print(f"Test Samples: {config['test_samples']}")
    print(f"LLM Analysis Limit: {config['llm_limit']}")
    print(f"LLM in Evaluation: {config['llm_in_eval']}")
    print("=" * 70)