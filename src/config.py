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


API_TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE4MDYxNTU1NDIsImtpZCI6IjgiLCJzY29wZXMiOlsiY2xvdWQ6cmVhZCIsImNsb3VkOndyaXRlIiwicmVwbzpyZWFkIiwicmVwbzp3cml0ZSJdLCJ2ZXIiOiJhcGk6MSIsInN1YiI6IjRiNTM4MGYyLWY1M2QtNDUwYS05M2E5LTgzMTdmYjJkZWFhMCJ9.PHMu6bLwPxSPFdv2o41n8BauSoCjtVnU2bg9ViSVSiZxLoyRdWBo6MN-_fMJfZyC4UvXuaIaLlfNCaKTVXzW6DDWvH7AodRTnP32XFBRzeaw-1QJ2_3sF_FrBDeSeEk1rhMNY87z6yHUO9AauQOT1A4hVw4WZJcs1IfMuVhQpYbrUlhwgb61JMuvdjOGKet1uITO_g_ZOmK2LJg-IX9Ek8NanqGmodMSwCarW6nvUSFxqkzyM6ekYYmJ2--5KMV5SqGPJGe1NNwDstCAb4fYK5adlRg9bcJve3Lf7IQB7kPSxvX4tlhyQywZOguM1As-knXSwIdk2qkO0vGwIpLvJw"
__all__ = [
    'CONNECT_ENDPOINT', 
    'NAVIGATOR_ENDPOINT', 
    'API_TOKEN',
    'DEMO_MODE',
    'LEGITIMATE_MERCHANTS',
    'SUSPICIOUS_MERCHANTS'
]


# Anaconda Connect - Production deployed model

CONNECT_ENDPOINT = "https://demo.se.sb.anacondaconnect.com/api/ai/inference/serve/bca8f194-702f-43d6-afa0-ec8ec306064a/v1/chat/completions"

# AI Navigator - Local LLM server
NAVIGATOR_ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"


# ================================================================================
# MODEL CONFIGURATION
# ================================================================================

# LLM Model
LLM_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LLM_MAX_NEW_TOKENS = 50
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