"""
Fraud Detection Models

This module contains:
- Hybrid ML+LLM fraud detector
- LLM text analysis functions
- Model evaluation utilities

Persona: Sarah Chen (Data Scientist), Marcus (ML Engineer)
Anaconda Value: Seamless integration of traditional ML + LLM
"""

import time
import re
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from transformers import AutoTokenizer, AutoModelForCausalLM
from .config import (
    LLM_MODEL_NAME, LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE,
    MODEL_WEIGHTS, XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_RANDOM_STATE,
    LOW_RISK_THRESHOLD
)


# ================================================================================
# LLM INITIALIZATION
# ================================================================================

# Global variables for lazy loading
_tokenizer = None
_model = None
_llm_cache = {}


def load_llm_model(verbose=True):
    """
    Load Qwen 2.5 7B model for text analysis
    
    Returns:
        Tuple of (tokenizer, model)
        
    Anaconda Value:
        - Desktop handles large model dependencies automatically
        - torch + transformers tracked in environment
        - Model weights cached after first load (~4.68GB)
        
    Performance:
        - First load: 2-3 minutes
        - Subsequent loads: <30 seconds (cached)
    """
    global _tokenizer, _model
    
    if _tokenizer is not None and _model is not None:
        if verbose:
            print("Using cached LLM model")
        return _tokenizer, _model
    
    if verbose:
        print(f"\n Loading {LLM_MODEL_NAME}...")
        print(f"  • Model size: ~4.68GB")
        print(f"  • First load: 2-3 minutes")
        print(f"  • Subsequent loads: <30 seconds (cached)")
    
    _tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_NAME,
        trust_remote_code=True
    )
    
    _model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    _model.eval()
    
    device = next(_model.parameters()).device
    
    if verbose:
        print(f"Qwen 2.5 7B loaded successfully")
        print(f"  • Device: {device}")
        print(f"  • Memory: ~4.88GB")
    
    return _tokenizer, _model


# ================================================================================
# LLM TEXT ANALYSIS
# ================================================================================

def analyze_merchant_llm(description, amount, use_cache=True):
    """
    Analyze merchant description for fraud indicators using LLM
    
    Args:
        description: Merchant name/description
        amount: Transaction amount
        use_cache: Use cached results for repeated queries
        
    Returns:
        float: Fraud probability (0.0 to 1.0)
        
    Optimization:
        - Result caching (avoid duplicate LLM calls)
        - Reduced token generation (5 vs 10)
        - Simplified prompt (~3x faster)
        
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
    
    # Load model if not already loaded
    tokenizer, model = load_llm_model(verbose=False)
    
    # Build concise prompt
    prompt = f"""Analyze: {description}, ${amount:.2f}
Fraud risk score (0.0-1.0):"""
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=LLM_MAX_NEW_TOKENS,
                temperature=LLM_TEMPERATURE,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract numeric score
        numbers = re.findall(r'\d+\.?\d*', response.split(":")[-1])
        score = float(numbers[0]) if numbers else 0.5
        
        # Clamp to valid range
        score = min(max(score, 0.0), 1.0)
        
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
        print(f"  • Stage 2: Qwen 2.5 7B (high-risk only)")
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
            
        Note: LLM requires no training (pre-trained foundation model)
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
            2. LLM analyzes high-risk cases (if descriptions provided)
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
            print(f"   Stage 2: Analyzing {high_risk_mask.sum()} high-risk cases with LLM...")
        
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
    Quick test of LLM analysis function
    
    Use Case: Verify LLM is working before running full pipeline
    """
    if verbose:
        print("\n Testing LLM analysis...")
    
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
        'llm_loaded': _model is not None,
        'llm_model_name': LLM_MODEL_NAME,
        'cache_size': len(_llm_cache),
        'device': str(next(_model.parameters()).device) if _model else 'Not loaded'
    }
    return info