"""
Fraud Detection Models

This module contains:
- Hybrid ML+LLM fraud detector
- LLM text analysis functions (via Anaconda Connect API)
- Model evaluation utilities

Persona: Sarah Chen (Data Scientist), Marcus (ML Engineer)
Anaconda Value: Seamless integration of traditional ML + LLM
"""

import time
import re
import json
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from .config import (
    LLM_MODEL_NAME, LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE,
    MODEL_WEIGHTS, XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_RANDOM_STATE,
    LOW_RISK_THRESHOLD, CONNECT_ENDPOINT, API_TOKEN
)


# ================================================================================
# LLM INITIALIZATION (API-based — no local model download required)
# ================================================================================

# Global variables
_session = None
_llm_cache = {}


def _get_session():
    """
    Get or create a reusable requests session for the Connect API.
    
    Returns:
        requests.Session configured with auth headers
    """
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {API_TOKEN}',
            'User-Agent': 'Anaconda-Fraud-Detection/1.0'
        })
    return _session


def load_llm_model(verbose=True):
    """
    Initialize connection to Meta-Llama-3.1-8B-Instruct via Anaconda Connect API.
    
    Unlike local model loading, this validates the API endpoint is reachable
    without downloading the ~4.68GB model weights.
    
    Returns:
        Tuple of (session, endpoint_url) — drop-in replacement for (tokenizer, model)
        
    Anaconda Value:
        - Model served via Anaconda Connect (AI Catalyst)
        - No local GPU or model download required
        - Same model, production-grade infrastructure
        
    Performance:
        - Initialization: <1 second (just an HTTP check)
        - Per-call latency: ~100-500ms (network round-trip)
    """
    session = _get_session()
    
    if verbose:
        print(f"\n Loading {LLM_MODEL_NAME}...")
        print(f"  • Endpoint: {CONNECT_ENDPOINT[:60]}...")
        print(f"  • Mode: API inference (no local model download)")
    
    # Validate endpoint is reachable
    try:
        test_payload = {
            "messages": [
                {"role": "user", "content": "test"}
            ],
            "max_tokens": 1,
            "temperature": 0.0
        }
        resp = session.post(CONNECT_ENDPOINT, json=test_payload, timeout=10)
        
        if resp.status_code == 200:
            if verbose:
                print(f" {LLM_MODEL_NAME} connected successfully")
                print(f"  • Status: API reachable")
                print(f"  • Auth: Valid")
        elif resp.status_code == 401:
            print(f"  WARNING: Authentication failed (401). Check API_TOKEN in config.py")
        elif resp.status_code == 403:
            print(f"  WARNING: Access forbidden (403). Check API permissions.")
        else:
            if verbose:
                print(f"  WARNING: Endpoint returned HTTP {resp.status_code}")
                print(f"  The model may still work — some endpoints reject minimal test payloads.")
                
    except requests.exceptions.ConnectionError:
        print(f"  WARNING: Cannot reach Anaconda Connect endpoint.")
        print(f"  LLM analysis will fall back to neutral scores (0.5).")
    except requests.exceptions.Timeout:
        print(f"  WARNING: Connection timed out.")
        print(f"  LLM analysis will fall back to neutral scores (0.5).")
    
    return session, CONNECT_ENDPOINT


# ================================================================================
# LLM TEXT ANALYSIS (via Anaconda Connect API)
# ================================================================================

def analyze_merchant_llm(description, amount, use_cache=True):
    """
    Analyze merchant description for fraud indicators using LLM via API.
    
    Args:
        description: Merchant name/description
        amount: Transaction amount
        use_cache: Use cached results for repeated queries
        
    Returns:
        float: Fraud probability (0.0 to 1.0)
        
    Optimization:
        - Result caching (avoid duplicate API calls)
        - Concise prompt for fast inference
        - Reusable HTTP session
        
    Business Logic:
        LLM analyzes text patterns that traditional ML might miss:
        - Suspicious keywords (BITCOIN, WIRE, CASINO)
        - Unusual merchant names
        - High-risk industries
    """
    global _llm_cache
    
    # Check cache first
    if use_cache:
        cache_key = f"{description}_{int(amount)}"
        if cache_key in _llm_cache:
            return _llm_cache[cache_key]
    
    # Ensure session is ready
    session = _get_session()
    
    # Build concise prompt
    prompt = f"""Analyze this transaction for fraud risk and respond with ONLY a number between 0.0 and 1.0:
Merchant: {description}
Amount: ${amount:.2f}
Fraud risk score (0.0=safe, 1.0=fraud):"""
    
    try:
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a fraud detection expert. Respond with ONLY a single decimal number between 0.0 and 1.0 representing the fraud risk score. No explanation."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": LLM_MAX_NEW_TOKENS,
            "temperature": LLM_TEMPERATURE
        }
        
        resp = session.post(CONNECT_ENDPOINT, json=payload, timeout=15)
        
        if resp.status_code == 200:
            data = resp.json()
            
            # Extract response text from OpenAI-compatible format
            response_text = ""
            if 'choices' in data and len(data['choices']) > 0:
                choice = data['choices'][0]
                if 'message' in choice:
                    response_text = choice['message'].get('content', '')
                elif 'text' in choice:
                    response_text = choice['text']
            elif 'content' in data:
                response_text = data['content']
            
            # Extract numeric score from response
            numbers = re.findall(r'\d+\.?\d*', response_text)
            score = float(numbers[0]) if numbers else 0.5
            
            # Clamp to valid range
            score = min(max(score, 0.0), 1.0)
        else:
            # API error — return neutral score
            score = 0.5
        
        # Cache result
        if use_cache:
            _llm_cache[cache_key] = score
        
        return score
        
    except Exception as e:
        print(f"LLM analysis error: {e}")
        return 0.5  # Neutral score on error


def clear_llm_cache():
    """Clear the LLM result cache"""
    global _llm_cache
    _llm_cache = {}


# ================================================================================
# HYBRID FRAUD DETECTOR
# ================================================================================

class OptimizedHybridDetector:
    
    def __init__(self, llm_threshold=None, max_llm_calls=None, 
                 weights=None, n_estimators=None, max_depth=None):
        """
        Initialize hybrid detector
        
        Args:
            llm_threshold: XGB score above which to trigger LLM (default: 0.3)
            max_llm_calls: Max LLM calls in evaluation (for demo speed)
            weights: Dict with 'xgb' and 'llm' weights (default: 0.6/0.4)
            n_estimators: Random forest trees (default: 100)
            max_depth: Tree depth (default: 12)
        """
        # Use config defaults if not specified
        self.llm_threshold = llm_threshold or LOW_RISK_THRESHOLD
        self.max_llm_calls = max_llm_calls
        self.weights = weights or MODEL_WEIGHTS
        
        # Initialize XGBoost (Random Forest)
        self.xgb = RandomForestClassifier(
            n_estimators=n_estimators or XGB_N_ESTIMATORS,
            max_depth=max_depth or XGB_MAX_DEPTH,
            random_state=XGB_RANDOM_STATE,
            n_jobs=-1
        )
        
        print(f"\n Hybrid detector initialized:")
        print(f"  • Stage 1: XGBoost ({n_estimators or XGB_N_ESTIMATORS} trees, fast)")
        print(f"  • Stage 2: {LLM_MODEL_NAME} via Anaconda Connect API (high-risk only)")
        print(f"  • LLM trigger: XGB score > {self.llm_threshold}")
        if max_llm_calls:
            print(f"  • LLM limit: {max_llm_calls} calls (demo mode)")
        print(f"  • Weights: XGB={self.weights['xgb']}, LLM={self.weights['llm']}")
    
    def fit(self, X, y, verbose=True):
        """
        Train the XGBoost component
        
        Args:
            X: Feature matrix
            y: Labels (0=legit, 1=fraud)
            verbose: Print training progress
            
        Returns:
            self (for method chaining)
            
        Note: LLM requires no training (pre-trained foundation model served via API)
        """
        if verbose:
            print("\n Training XGBoost...")
        
        start = time.time()
        self.xgb.fit(X, y)
        elapsed = time.time() - start
        
        if verbose:
            print(f" Training complete in {elapsed:.2f}s")
        
        return self
    
    def predict_proba(self, X, descriptions=None, amounts=None, verbose=True):
        """
        Predict fraud probabilities for transactions
        
        Args:
            X: Feature matrix
            descriptions: Merchant descriptions (optional, for LLM)
            amounts: Transaction amounts (optional, for LLM)
            verbose: Print progress
            
        Returns:
            numpy array of fraud probabilities
            
        Two-Stage Process:
            1. XGBoost screens all transactions (fast)
            2. LLM analyzes high-risk cases via API (if descriptions provided)
        """
        if verbose:
            print(f"\n Analyzing {len(X):,} transactions...")
        
        # Stage 1: XGBoost screening (all transactions)
        xgb_probas = self.xgb.predict_proba(X)[:, 1]
        
        if verbose:
            print(f"   Stage 1: XGBoost screened all transactions")
        
        # If no descriptions, return XGBoost-only predictions
        if descriptions is None or amounts is None:
            return xgb_probas
        
        # Stage 2: LLM analysis for high-risk cases
        final_probas = xgb_probas.copy()
        high_risk_mask = xgb_probas > self.llm_threshold
        high_risk_count = high_risk_mask.sum()
        
        # Apply LLM limit if in demo mode
        if self.max_llm_calls and high_risk_count > self.max_llm_calls:
            high_risk_indices = np.where(high_risk_mask)[0]
            # Prioritize highest XGB scores
            top_indices = high_risk_indices[
                np.argsort(xgb_probas[high_risk_mask])[-self.max_llm_calls:]
            ]
            high_risk_mask = np.zeros(len(X), dtype=bool)
            high_risk_mask[top_indices] = True
            
            if verbose:
                print(f"   Limited LLM analysis to top {self.max_llm_calls} cases (demo mode)")
        
        llm_analyzed = 0
        if verbose and high_risk_mask.sum() > 0:
            print(f"   Stage 2: Analyzing {high_risk_mask.sum()} high-risk cases with LLM API...")
        
        start_time = time.time()
        for idx in np.where(high_risk_mask)[0]:
            llm_score = analyze_merchant_llm(
                descriptions.iloc[idx],
                amounts.iloc[idx]
            )
            # Weighted combination
            final_probas[idx] = (
                self.weights['xgb'] * xgb_probas[idx] + 
                self.weights['llm'] * llm_score
            )
            llm_analyzed += 1
        
        if verbose and llm_analyzed > 0:
            elapsed = time.time() - start_time
            print(f"  Stage 2: Analyzed {llm_analyzed} cases in {elapsed:.1f}s")
            print(f"  • Average: {elapsed/llm_analyzed:.2f}s per LLM call")
        
        return final_probas
    
    def predict(self, X, descriptions=None, amounts=None, threshold=0.5, verbose=True):
        """
        Predict fraud labels (0 or 1)
        
        Args:
            X: Feature matrix
            descriptions: Merchant descriptions (optional)
            amounts: Transaction amounts (optional)
            threshold: Decision threshold (default: 0.5)
            verbose: Print progress
            
        Returns:
            numpy array of predictions (0=legit, 1=fraud)
        """
        probas = self.predict_proba(X, descriptions, amounts, verbose)
        return (probas > threshold).astype(int)
    
    def get_feature_importance(self, feature_names=None, top_n=10):
        """
        Get feature importance from XGBoost component
        
        Args:
            feature_names: List of feature names (optional)
            top_n: Number of top features to return
            
        Returns:
            pandas DataFrame with feature importances
        """
        import pandas as pd
        
        importances = self.xgb.feature_importances_
        
        if feature_names is None:
            feature_names = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return df.head(top_n)


# ================================================================================
# MODEL TESTING UTILITIES
# ================================================================================

def test_llm_analysis(verbose=True):
    """
    Quick test of LLM analysis function via API
    
    Use Case: Verify API endpoint is working before running full pipeline
    """
    if verbose:
        print("\n Testing LLM analysis (via Anaconda Connect API)...")
    
    test_cases = [
        ("AMAZON.COM MKTP US", 67.89, "Low Risk"),
        ("BITCOIN ATM UNKNOWN", 3456.78, "High Risk"),
    ]
    
    for desc, amt, expected in test_cases:
        score = analyze_merchant_llm(desc, amt)
        risk = "HIGH" if score > 0.7 else "MEDIUM" if score > 0.3 else "LOW"
        
        if verbose:
            print(f"  • {desc}: {score:.2f} ({risk} risk) - Expected: {expected}")
    
    if verbose:
        print(" LLM analysis test complete")


def get_model_info():
    """
    Get information about loaded models
    
    Returns:
        dict with model information
    """
    info = {
        'llm_loaded': _session is not None,
        'llm_model_name': LLM_MODEL_NAME,
        'llm_mode': 'Anaconda Connect API',
        'cache_size': len(_llm_cache),
        'endpoint': CONNECT_ENDPOINT[:60] + '...'
    }
    return info