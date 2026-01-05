"""
Production API Clients for Deployed Models

This module provides clients for:
- Anaconda Connect deployed models
- AI Navigator local inference
- Fallback/mock predictions

Persona: Marcus (ML Engineer), Michael (VP Fraud Prevention)
Anaconda Value: AI Catalyst enables production deployment with auto-generated APIs
"""

import time
import json
import numpy as np
import requests
from datetime import datetime
from .config import CONNECT_ENDPOINT, NAVIGATOR_ENDPOINT, LEGITIMATE_MERCHANTS, SUSPICIOUS_MERCHANTS


# ================================================================================
# UNIFIED FRAUD DETECTION API CLIENT
# ================================================================================

class FraudDetectionAPI:
    """
    Multi-endpoint fraud detection API client with automatic fallback
    
    Priority Order:
        1. Anaconda Connect (Production deployed model)
        2. AI Navigator (Local LLM server)
        3. Mock Model (Heuristic fallback)
        
    Anaconda Value:
        - AI Catalyst auto-generates production API
        - Seamless deployment from notebook to production
        - Built-in load balancing and scaling
    """
    
    def __init__(self, connect_endpoint=None, navigator_endpoint=None):
        """
        Initialize API client
        
        Args:
            connect_endpoint: Anaconda Connect model URL
            navigator_endpoint: Local AI Navigator URL
        """
        self.connect_endpoint = connect_endpoint or CONNECT_ENDPOINT
        self.navigator_endpoint = navigator_endpoint or NAVIGATOR_ENDPOINT
        self.session = requests.Session()
        self.last_source = "Not Used Yet"
        
        print(f"\n📡 API Client initialized")
        print(f"  • Connect: {self.connect_endpoint[:60]}...")
        print(f"  • Navigator: {self.navigator_endpoint}")
    
    def predict(self, merchant, amount, features=None):
        """
        Predict fraud for a single transaction with automatic fallback
        
        Args:
            merchant: Merchant description
            amount: Transaction amount
            features: Optional feature vector (generated if not provided)
            
        Returns:
            dict with keys:
                - success: bool
                - prediction: int (0=legit, 1=fraud)
                - probability: float (0.0 to 1.0)
                - latency_ms: float
                - timestamp: datetime
                - source: str
        """
        if features is None:
            features = self._generate_features(merchant, amount)
        
        start_time = time.time()
        
        # 1) Try Anaconda Connect first
        result = self._try_connect_inference(merchant, amount, features)
        if result is not None:
            result["latency_ms"] = (time.time() - start_time) * 1000
            result["timestamp"] = datetime.now()
            result["source"] = "Anaconda Connect (Deployed Model)"
            self.last_source = result["source"]
            return result
        
        # 2) Fallback to AI Navigator
        result = self._try_navigator_llm(merchant, amount)
        if result is not None:
            result["latency_ms"] = (time.time() - start_time) * 1000
            result["timestamp"] = datetime.now()
            result["source"] = "AI Navigator (Local)"
            self.last_source = result["source"]
            return result
        
        # 3) Final fallback: Mock model
        latency = (time.time() - start_time) * 1000
        result = self._mock_predict(merchant, amount, features, latency)
        self.last_source = result["source"]
        return result
    
    # ---------- Adapter 1: Anaconda Connect ----------
    
    def _try_connect_inference(self, merchant, amount, features):
        """
        Call Anaconda Connect deployed model
        
        Expected Response:
            {
                "prediction": [0 or 1],
                "probability": [0.0 to 1.0]
            }
        """
        payload = {
            "data": [features.tolist()],
            "merchant_description": [merchant],
            "amount": [float(amount)]
        }
        
        try:
            resp = self.session.post(
                self.connect_endpoint,
                json=payload,
                timeout=10
            )
            
            if resp.status_code != 200:
                return None
            
            result = resp.json()
            
            # Normalize output
            prob = result.get("probability", [0.5])[0]
            pred = result.get("prediction", [1 if prob >= 0.5 else 0])[0]
            
            return {
                "success": True,
                "prediction": int(pred),
                "probability": float(prob)
            }
            
        except Exception as e:
            # Silently fall through to next option
            return None
    
    # ---------- Adapter 2: AI Navigator (Local LLM) ----------
    
    def _try_navigator_llm(self, merchant, amount):
        """
        Call local AI Navigator chat completions endpoint
        
        Expected Response (OpenAI-compatible):
            {
                "choices": [{
                    "message": {
                        "content": '{"probability": 0.xx}'
                    }
                }]
            }
        """
        prompt = (
            "You are a fraud risk scorer. "
            "Return ONLY valid JSON with a single key 'probability' (0 to 1). "
            "No extra text.\n\n"
            f"Merchant: {merchant}\n"
            f"Amount: {float(amount):.2f}\n"
        )
        
        payload = {
            "messages": [
                {"role": "system", "content": "Return only JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0
        }
        
        try:
            resp = self.session.post(
                self.navigator_endpoint,
                json=payload,
                timeout=10
            )
            
            if resp.status_code != 200:
                return None
            
            out = resp.json()
            content = out["choices"][0]["message"]["content"]
            
            # Parse JSON response
            data = json.loads(content)
            prob = float(data["probability"])
            prob = max(0.0, min(1.0, prob))
            pred = 1 if prob >= 0.5 else 0
            
            return {
                "success": True,
                "prediction": pred,
                "probability": prob
            }
            
        except Exception as e:
            # Fall through to mock
            return None
    
    # ---------- Fallback: Mock Model (FIXED FOR REALISTIC SCORES) ----------
    
    def _mock_predict(self, merchant, amount, features, latency):
        """
        Heuristic-based mock predictions with realistic scoring
        
        IMPORTANT FIX: Now properly differentiates between:
        - Known legitimate merchants (WALMART, AMAZON) → Low scores (0.05-0.25)
        - Suspicious merchants (BITCOIN, CASINO) → High scores (0.65-0.95)
        - Unknown merchants → Medium scores (0.30-0.60)
        
        Use Case: Demo continues even when APIs are unavailable
        
        Logic:
            1. Check if merchant is in LEGITIMATE_MERCHANTS list
            2. Check for suspicious keywords
            3. Adjust for amount
            4. Add small random variation
        """
        merchant_upper = merchant.upper()
        
        # Define suspicious keywords
        suspicious_keywords = [
            'BITCOIN', 'CRYPTO', 'CASINO', 'WIRE', 
            'FOREIGN', 'UNKNOWN', 'UNVERIFIED', 'GAMBLING',
            'ATM UNKNOWN', 'DARK WEB', 'ANONYMOUS'
        ]
        
        # Check merchant type
        is_suspicious = any(k in merchant_upper for k in suspicious_keywords)
        is_known_legitimate = merchant in LEGITIMATE_MERCHANTS
        
        # Determine base probability based on merchant type
        if is_suspicious:
            # High-risk merchants: crypto, casinos, wire transfers
            base_prob = 0.72
            
        elif is_known_legitimate:
            # Known legitimate merchants: Amazon, Walmart, Target, etc.
            base_prob = 0.08  # Very low base score
            
        else:
            # Unknown merchants (not in either list)
            base_prob = 0.35  # Medium-low score
        
        # Amount-based adjustment
        if is_known_legitimate:
            # For legitimate merchants, amount has minimal impact
            if amount > 2000:
                amount_factor = 0.08  # Even high amounts stay relatively low
            elif amount > 1000:
                amount_factor = 0.05
            elif amount > 500:
                amount_factor = 0.02
            else:
                amount_factor = 0.0
        else:
            # For suspicious/unknown merchants, amount matters more
            if amount > 3000:
                amount_factor = 0.18
            elif amount > 2000:
                amount_factor = 0.15
            elif amount > 1000:
                amount_factor = 0.10
            elif amount > 500:
                amount_factor = 0.05
            else:
                amount_factor = 0.0
        
        # Combine factors with small random variation
        probability = base_prob + amount_factor + np.random.uniform(-0.03, 0.03)
        
        # Apply realistic bounds based on merchant type
        if is_known_legitimate:
            # Legitimate merchants: Keep scores low (0.05-0.35 max)
            probability = min(max(probability, 0.05), 0.35)
            
        elif is_suspicious:
            # Suspicious merchants: Keep scores high (0.60-0.98)
            probability = min(max(probability, 0.60), 0.98)
            
        else:
            # Unknown merchants: Medium range (0.25-0.65)
            probability = min(max(probability, 0.25), 0.65)
        
        # Final safety bounds
        probability = min(max(probability, 0.01), 0.99)
        
        # Determine prediction
        prediction = 1 if probability > 0.5 else 0
        
        return {
            "success": True,
            "prediction": prediction,
            "probability": float(probability),
            "latency_ms": latency,
            "timestamp": datetime.now(),
            "source": "Mock Model (Fallback)"
        }
    
    def _generate_features(self, merchant, amount):
        """
        Generate synthetic feature vector for API calls
        
        Returns:
            numpy array of 30 features
        """
        np.random.seed(int(time.time() * 1000) % 2**32)
        
        # High variance for suspicious merchants
        if any(s in merchant.upper() for s in ["BITCOIN", "CRYPTO", "CASINO", "WIRE", "FOREIGN"]):
            features = np.random.randn(28) * 3
        else:
            features = np.random.randn(28) * 0.5
        
        # Add Time and Amount
        features = np.append(features, [np.random.randint(0, 172800), amount])
        return features
    
    def test_connection(self):
        """
        Test connectivity to all endpoints
        
        Returns:
            dict with connection status for each endpoint
        """
        print("\n🔍 Testing API connections...")
        
        results = {
            'connect': False,
            'navigator': False,
            'mock': True  # Always available
        }
        
        # Test Connect
        try:
            test_payload = {
                "data": [[0] * 30],
                "merchant_description": ["TEST"],
                "amount": [100.0]
            }
            resp = self.session.post(
                self.connect_endpoint,
                json=test_payload,
                timeout=5
            )
            results['connect'] = resp.status_code in [200, 400]
            print(f"  ✓ Anaconda Connect: {'Online' if results['connect'] else 'Offline'}")
        except Exception as e:
            print(f"  ✗ Anaconda Connect: Offline")
        
        # Test Navigator
        try:
            test_payload = {
                "messages": [{"role": "user", "content": "test"}],
                "temperature": 0.0
            }
            resp = self.session.post(
                self.navigator_endpoint,
                json=test_payload,
                timeout=5
            )
            results['navigator'] = resp.status_code in [200, 400]
            print(f"  ✓ AI Navigator: {'Online' if results['navigator'] else 'Offline'}")
        except Exception as e:
            print(f"  ✗ AI Navigator: Offline")
        
        print(f"  ✓ Mock Model: Always available (fallback)")
        
        return results


# ================================================================================
# LEGACY PRODUCTION API CLIENT
# ================================================================================

class ProductionFraudAPI:
    """
    Legacy production API client (for backward compatibility)
    
    This is the original single-endpoint client.
    Prefer FraudDetectionAPI for new code (has automatic fallback).
    """
    
    def __init__(self, endpoint=None):
        """
        Initialize production API client
        
        Args:
            endpoint: API endpoint URL (default: from config)
        """
        self.endpoint = endpoint or CONNECT_ENDPOINT
        self.session = requests.Session()
        print(f"\n✓ Legacy API Client initialized")
        print(f"  • Endpoint: {self.endpoint[:60]}...")
    
    def test_connection(self):
        """Test if API is accessible"""
        try:
            test_payload = {
                "data": [[0] * 30],
                "merchant_description": ["TEST CONNECTION"],
                "amount": [100.0]
            }
            response = self.session.post(
                self.endpoint,
                json=test_payload,
                timeout=10
            )
            return response.status_code in [200, 400]
        except Exception as e:
            print(f"  ⚠️  Connection test error: {str(e)}")
            return False
    
    def predict_single(self, transaction_features, merchant_desc, amount):
        """
        Predict fraud for a single transaction
        
        Args:
            transaction_features: Feature vector
            merchant_desc: Merchant description
            amount: Transaction amount
            
        Returns:
            dict with prediction results
        """
        payload = {
            "data": [transaction_features.tolist()],
            "merchant_description": [merchant_desc],
            "amount": [float(amount)]
        }
        
        start_time = time.time()
        
        try:
            response = self.session.post(
                self.endpoint,
                json=payload,
                timeout=30
            )
            
            latency = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'prediction': result.get('prediction', [0])[0],
                    'probability': result.get('probability', [0.0])[0],
                    'latency_ms': latency,
                    'source': 'PRODUCTION API'
                }
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}",
                    'latency_ms': latency
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'latency_ms': -1
            }


# ================================================================================
# BATCH PREDICTION UTILITIES
# ================================================================================

def batch_predict(api_client, transactions_df, batch_size=10, verbose=True):
    """
    Predict fraud for multiple transactions
    
    Args:
        api_client: FraudDetectionAPI instance
        transactions_df: DataFrame with 'merchant' and 'amount' columns
        batch_size: Process in batches (for rate limiting)
        verbose: Print progress
        
    Returns:
        list of prediction results
    """
    results = []
    total = len(transactions_df)
    
    if verbose:
        print(f"\n🔄 Processing {total} transactions in batches of {batch_size}...")
    
    for i in range(0, total, batch_size):
        batch = transactions_df.iloc[i:i+batch_size]
        
        for idx, row in batch.iterrows():
            result = api_client.predict(row['merchant'], row['amount'])
            results.append(result)
        
        if verbose and i % (batch_size * 5) == 0:
            print(f"  • Processed {min(i+batch_size, total)}/{total}")
        
        # Small delay to respect rate limits
        time.sleep(0.1)
    
    if verbose:
        successful = sum(1 for r in results if r['success'])
        print(f"✓ Complete: {successful}/{total} successful predictions")
    
    return results