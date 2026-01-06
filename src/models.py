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
            print("✓ Using cached LLM model")
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
        print(f"✓ Qwen 2.5 7B loaded successfully")
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
        print(f"⚠️  LLM analysis error: {e}")
        return 0.5  # Neutral score on error


def clear_llm_cache():
    """Clear the LLM result cache"""
    global _llm_cache
    _llm_cache = {}


# ================================================================================
# HYBRID FRAUD DETECTOR
# ================================================================================

class OptimizedHybridDetector:
    """
    Two-stage fraud detection system
    You're absolutely right - I apologize for the confusion! Let me create the actual VISUAL presentation that will render immediately for you to use. This will open as an interactive presentation you can click through right now!
<artifact identifier="anaconda-fraud-presentation-visual" type="application/vnd.ant.react" title="Anaconda Fraud Detection - Interactive Presentation">
import React, { useState, useEffect } from 'react';
import { ChevronLeft, ChevronRight, Home, Shield, TrendingUp, Cloud, Zap, DollarSign, CheckCircle, AlertCircle, Cpu, Database, Box, Lock, Rocket, Award, BarChart3, Server, GitBranch, Package, FileCode, Target, Users, Calendar, Eye, Layers, ArrowRight, Check, X, Terminal } from 'lucide-react';
const FraudPresentation = () => {
const [currentSlide, setCurrentSlide] = useState(0);
const totalSlides = 32;
const nextSlide = () => currentSlide < totalSlides - 1 && setCurrentSlide(currentSlide + 1);
const prevSlide = () => currentSlide > 0 && setCurrentSlide(currentSlide - 1);
useEffect(() => {
const handleKeyDown = (e) => {
if (e.key === 'ArrowRight') nextSlide();
if (e.key === 'ArrowLeft') prevSlide();
if (e.key === 'Home') setCurrentSlide(0);
};
window.addEventListener('keydown', handleKeyDown);
return () => window.removeEventListener('keydown', handleKeyDown);
}, [currentSlide]);
const Header = ({ num }) => (
<div className="flex justify-between items-center mb-6 pb-4 border-b-4 border-green-600">
<div className="flex items-center gap-3 text-green-600 text-2xl font-bold">
<div className="w-10 h-10 bg-green-600 rounded-full flex items-center justify-center text-white text-xl">A</div>
ANACONDA
</div>
<div className="text-gray-500 font-semibold text-lg">Slide {num} / {totalSlides}</div>
</div>
);
const slides = [];
// SLIDE 0 - Title
slides.push(
<div className="h-full flex items-center justify-center bg-gradient-to-br from-green-600 to-green-800 text-white p-16">
<div className="text-center max-w-5xl">
<Shield className="w-40 h-40 mx-auto mb-10 drop-shadow-2xl animate-pulse" />
<h1 className="text-8xl font-bold mb-8 drop-shadow-lg">Enterprise Fraud Detection</h1>
<h2 className="text-5xl font-light mb-10">on Anaconda Platform</h2>
<div className="mt-12 p-10 bg-white/10 backdrop-blur-xl rounded-2xl border-2 border-white/30 shadow-2xl">
<p className="text-3xl leading-relaxed mb-4">Building Production-Ready AI</p>
<p className="text-3xl leading-relaxed">from Notebook to Deployment</p>
<p className="text-xl mt-6 text-green-200">Self-Hosted Core • AI Catalyst • Partner Ecosystem</p>
</div>
</div>
</div>
);
// SLIDE 1 - The $50M Question
slides.push(
<div className="h-full flex flex-col p-12 bg-white">
<Header num={1} />
<div className="flex-1 flex flex-col justify-center">
<h1 className="text-6xl font-bold mb-10 text-center text-gray-900">The $50M Question</h1>
<div className="text-4xl text-gray-700 italic text-center mb-16 max-w-5xl mx-auto leading-snug">
"How do we go from fraud detection notebook to production API that prevents $50M in annual fraud losses?"
</div>
    <div className="grid grid-cols-2 gap-8 max-w-6xl mx-auto">
      <div className="bg-red-50 border-4 border-red-500 rounded-xl p-8">
        <div className="flex items-center gap-3 mb-4">
          <X className="w-10 h-10 text-red-600" />
          <h3 className="text-3xl font-bold text-red-600">Traditional Answer</h3>
        </div>
        <div className="space-y-4 text-lg">
          <div className="bg-white p-4 rounded-lg">
            <div className="font-bold text-xl mb-2">Timeline</div>
            <div className="text-3xl font-bold text-red-600">24-48 weeks</div>
          </div>
          <div className="bg-white p-4 rounded-lg">
            <div className="font-bold text-xl mb-2">Cost</div>
            <div className="text-3xl font-bold text-red-600">$1M-$2M</div>
          </div>
          <div className="bg-white p-4 rounded-lg">
            <div className="font-bold text-xl mb-2">Success Rate</div>
            <div className="text-3xl font-bold text-red-600">20%</div>
          </div>
        </div>
      </div>

      <div className="bg-green-50 border-4 border-green-600 rounded-xl p-8">
        <div className="flex items-center gap-3 mb-4">
          <CheckCircle className="w-10 h-10 text-green-600" />
          <h3 className="text-3xl font-bold text-green-600">Anaconda Answer</h3>
        </div>
        <div className="space-y-4 text-lg">
          <div className="bg-white p-4 rounded-lg">
            <div className="font-bold text-xl mb-2">Timeline</div>
            <div className="text-3xl font-bold text-green-600">4-8 weeks</div>
          </div>
          <div className="bg-white p-4 rounded-lg">
            <div className="font-bold text-xl mb-2">Cost</div>
            <div className="text-3xl font-bold text-green-600">$300K</div>
          </div>
          <div className="bg-white p-4 rounded-lg">
            <div className="font-bold text-xl mb-2">Success Rate</div>
            <div className="text-3xl font-bold text-green-600">95%</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
);
// SLIDE 2 - Three Enterprise Challenges
slides.push(
<div className="h-full flex flex-col p-12 bg-white">
<Header num={2} />
<h1 className="text-5xl font-bold mb-8 text-gray-900">The Enterprise Fraud Detection Problem</h1>
<h2 className="text-3xl text-gray-600 mb-10">Three Fundamental Challenges</h2>
  <div className="grid grid-cols-1 gap-6">
    <div className="bg-red-50 border-l-8 border-red-500 p-6 rounded-r-xl">
      <div className="flex items-start gap-4">
        <div className="bg-red-500 text-white rounded-full w-12 h-12 flex items-center justify-center text-2xl font-bold flex-shrink-0">1</div>
        <div>
          <h3 className="text-2xl font-bold text-red-600 mb-3">The "Last Mile" Problem</h3>
          <div className="text-lg space-y-2">
            <p className="font-semibold">Data Scientists: "Our fraud model works perfectly in Jupyter!"</p>
            <p className="font-semibold">Engineering Team: "Great. Now we need 6 months to productionize it."</p>
            <div className="mt-4 p-4 bg-white rounded-lg">
              <p className="text-red-700 font-bold">Reality: Model is outdated by production launch</p>
              <p className="text-gray-700">Cost: $500K-$2M wasted + $10M-$50M in missed fraud prevention</p>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div className="bg-orange-50 border-l-8 border-orange-500 p-6 rounded-r-xl">
      <div className="flex items-start gap-4">
        <div className="bg-orange-500 text-white rounded-full w-12 h-12 flex items-center justify-center text-2xl font-bold flex-shrink-0">2</div>
        <div>
          <h3 className="text-2xl font-bold text-orange-600 mb-3">The Security Supply Chain Crisis</h3>
          <div className="grid grid-cols-2 gap-4 text-lg mt-4">
            <div className="bg-white p-4 rounded-lg">
              <p className="font-semibold mb-2">PyTorch + Transformers + XGBoost</p>
              <p className="text-3xl font-bold text-orange-600">500+ packages</p>
            </div>
            <div className="bg-white p-4 rounded-lg">
              <p className="font-semibold mb-2">Total Dependencies</p>
              <p className="text-3xl font-bold text-orange-600">5,000+</p>
            </div>
          </div>
          <div className="mt-4 p-4 bg-white rounded-lg">
            <p className="font-bold text-orange-700">Question: How many have CVEs? Who's liable?</p>
            <p className="text-gray-700">Answer: Unknown → $2M-$10M breach risk</p>
          </div>
        </div>
      </div>
    </div>

    <div className="bg-purple-50 border-l-8 border-purple-500 p-6 rounded-r-xl">
      <div className="flex items-start gap-4">
        <div className="bg-purple-500 text-white rounded-full w-12 h-12 flex items-center justify-center text-2xl font-bold flex-shrink-0">3</div>
        <div>
          <h3 className="text-2xl font-bold text-purple-600 mb-3">The Integration Nightmare</h3>
          <div className="text-lg space-y-2">
            <p><strong>Data Team:</strong> "We use Snowflake for transaction data"</p>
            <p><strong>ML Team:</strong> "We use Databricks for feature engineering"</p>
            <p><strong>DevOps:</strong> "We use Docker/Kubernetes"</p>
            <p><strong>Cloud Team:</strong> "We're on AWS... but might move to Azure"</p>
            <div className="mt-4 p-4 bg-white rounded-lg">
              <p className="font-bold text-purple-700">Result: 5 vendors, 12-18 months integration</p>
              <p className="text-gray-700">Cost: $1M-$3M in custom glue code</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
);
// SLIDE 3 - Why This Demo Matters
slides.push(
<div className="h-full flex flex-col p-12 bg-white">
<Header num={3} />
<h1 className="text-5xl font-bold mb-6 text-gray-900">Why This Fraud Detection Demo Matters</h1>
<h2 className="text-3xl text-gray-600 mb-8">One Demo, Seven Strategic Value Propositions</h2>
  <div className="grid grid-cols-1 gap-4 overflow-y-auto">
    {[
      { icon: Rocket, title: "Notebook-to-Production in Days", value: "AI Catalyst: Upload notebook → Production API (24 hours)", pain: '"Our models never make it to production"' },
      { icon: Layers, title: "Hybrid ML+LLM Architecture", value: "XGBoost (speed) + Qwen 2.5 7B (explainability)", pain: '"Regulators demand explanations, not scores"' },
      { icon: Shield, title: "Supply Chain Security", value: "CVE-scanned PyTorch + 500 deps + $10M indemnification", pain: '"One vulnerable package = $2M-$10M breach"' },
      { icon: Cloud, title: "Partner Ecosystem Integration", value: "Snowflake + Databricks + Docker + AWS/Azure/GCP + NVIDIA", pain: '"We\'re stuck choosing between vendors"' },
      { icon: Target, title: "Interactive Demonstration", value: "Live fraud analysis with widgets + API testing", pain: '"Can we test before buying?"' },
      { icon: DollarSign, title: "Complete ROI Modeling", value: "$53M annual value vs $300K cost = 17,000% ROI", pain: '"What\'s the business case?"' },
      { icon: Globe, title: "Multi-Cloud Flexibility", value: "Same fraud detection on AWS/Azure/GCP/on-prem", pain: '"We might change clouds"' }
    ].map((item, idx) => (
      <div key={idx} className="flex items-center gap-6 bg-gradient-to-r from-green-50 to-white border-l-4 border-green-600 p-5 rounded-r-lg shadow-sm hover:shadow-md transition">
        <div className="bg-green-600 text-white rounded-full w-16 h-16 flex items-center justify-center flex-shrink-0">
          <item.icon className="w-8 h-8" />
        </div>
        <div className="flex-1">
          <h3 className="text-xl font-bold text-gray-900 mb-2">{item.title}</h3>
          <p className="text-gray-700 mb-1">{item.value}</p>
          <p className="text-sm text-gray-500 italic">{item.pain}</p>
        </div>
        <div className="bg-green-600 text-white rounded-full w-10 h-10 flex items-center justify-center font-bold text-xl flex-shrink-0">
          {idx + 1}
        </div>
      </div>
    ))}
  </div>
</div>
);
// SLIDE 4 - Solution Architecture
slides.push(
<div className="h-full flex flex-col p-12 bg-white">
<Header num={4} />
<h1 className="text-5xl font-bold mb-6 text-gray-900">Complete Solution Architecture</h1>
<h2 className="text-3xl text-gray-600 mb-8">Anaconda Platform: Enterprise Foundation for Production AI</h2>
  <div className="flex flex-col gap-6">
    <div className="bg-blue-50 border-4 border-blue-500 rounded-xl p-6">
      <div className="flex items-center gap-3 mb-4">
        <FileCode className="w-8 h-8 text-blue-600" />
        <h3 className="text-2xl font-bold text-blue-600">Development Layer (Anaconda Notebooks)</h3>
      </div>
      <div className="grid grid-cols-5 gap-3 text-center">
        {['01: Data Exploration', '02: Model Training', '03: Evaluation', '04: Business Impact', '05: Interactive Demo'].map((nb, i) => (
          <div key={i} className="bg-white p-4 rounded-lg border-2 border-blue-300">
            <div className="font-bold text-blue-600 mb-2">Notebook {nb.split(':')[0]}</div>
            <div className="text-sm text-gray-600">{nb.split(':')[1]}</div>
          </div>
        ))}
      </div>
      <div className="mt-4 p-4 bg-white rounded-lg">
        <p className="font-semibold text-gray-700">Powered by: Anaconda Self-Hosted Core</p>
        <p className="text-sm text-gray-600">CVE-scanned PyTorch, XGBoost, transformers • $10M legal indemnification</p>
      </div>
    </div>

    <div className="flex justify-center">
      <ArrowRight className="w-12 h-12 text-gray-400" />
    </div>

    <div className="bg-purple-50 border-4 border-purple-500 rounded-xl p-6">
      <div className="flex items-center gap-3 mb-4">
        <Server className="w-8 h-8 text-purple-600" />
        <h3 className="text-2xl font-bold text-purple-600">Partner Ecosystem Layer</h3>
      </div>
      <div className="grid grid-cols-3 gap-4">
        {[
          { name: 'Snowflake', desc: 'Transaction History' },
          { name: 'Databricks', desc: 'Feature Engineering' },
          { name: 'Docker', desc: 'Containers' },
          { name: 'AWS/Azure/GCP', desc: 'Cloud Infrastructure' },
          { name: 'NVIDIA', desc: 'GPU Acceleration' },
          { name: 'PyTorch', desc: 'LLM Framework' }
        ].map((partner, i) => (
          <div key={i} className="bg-white p-4 rounded-lg border-2 border-purple-300 text-center">
            <div className="font-bold text-purple-600">{partner.name}</div>
            <div className="text-sm text-gray-600 mt-1">{partner.desc}</div>
          </div>
        ))}
      </div>
    </div>

    <div className="flex justify-center">
      <ArrowRight className="w-12 h-12 text-gray-400" />
    </div>

    <div className="bg-green-50 border-4 border-green-600 rounded-xl p-6">
      <div className="flex items-center gap-3 mb-4">
        <Rocket className="w-8 h-8 text-green-600" />
        <h3 className="text-2xl font-bold text-green-600">Production Deployment (AI Catalyst)</h3>
      </div>
      <div className="grid grid-cols-4 gap-4">
        {['REST API Endpoint', 'Auto-Scaling (1-10K TPS)', 'Monitoring Dashboards', 'Model Versioning'].map((feature, i) => (
          <div key={i} className="bg-white p-4 rounded-lg border-2 border-green-300 text-center">
            <CheckCircle className="w-6 h-6 text-green-600 mx-auto mb-2" />
            <div className="text-sm font-semibold text-gray-700">{feature}</div>
          </div>
        ))}
      </div>
      <div className="mt-4 grid grid-cols-2 gap-4">
        <div className="bg-green-100 p-3 rounded-lg text-center">
          <div className="text-sm font-semibold text-gray-700">Deployment Time</div>
          <div className="text-2xl font-bold text-green-600">24 hours</div>
        </div>
        <div className="bg-green-100 p-3 rounded-lg text-center">
          <div className="text-sm font-semibold text-gray-700">Production SLA</div>
          <div className="text-2xl font-bold text-green-600">&lt;200ms latency</div>
        </div>
      </div>
    </div>
  </div>
</div>
);
// SLIDE 5 - Demo Structure
slides.push(
<div className="h-full flex flex-col p-12 bg-white">
<Header num={5} />
<h1 className="text-5xl font-bold mb-6 text-gray-900">Demo Structure Overview</h1>
<h2 className="text-3xl text-gray-600 mb-8">Five Notebooks = Complete Fraud Detection Journey</h2>
  <div className="grid grid-cols-1 gap-5">
    {[
      { num: '01', icon: Database, title: 'Data Exploration', subtitle: 'Foundation', what: 'Analyze 284K transactions, identify 0.17% fraud rate', why: 'Proves model handles realistic class imbalance', value: 'Establishes fraud detection feasibility', color: 'blue' },
      { num: '02', icon: Cpu, title: 'Model Training', subtitle: 'Intelligence', what: 'Train XGBoost + Qwen 2.5 7B (96% recall, 95% precision)', why: 'Demonstrates production-grade accuracy + explainability', value: '$53M annual fraud prevention validated', color: 'purple' },
      { num: '03', icon: BarChart3, title: 'Advanced Evaluation', subtitle: 'Optimization', what: 'ROC curves, precision-recall, threshold tuning', why: 'Proves model is optimized, not just "good enough"', value: 'Threshold optimization = $1M+ additional value', color: 'indigo' },
      { num: '04', icon: DollarSign, title: 'Business Impact', subtitle: 'ROI', what: 'Calculate $53M annual value (fraud + ops savings)', why: 'Translates technical metrics into CFO language', value: '17,000% ROI justifies platform investment', color: 'green' },
      { num: '05', icon: Target, title: 'Interactive Demo', subtitle: 'Proof', what: 'Live fraud detection with widgets + API + batch', why: 'Lets prospects test fraud detection themselves', value: 'Hands-on validation = 40-60% higher close rates', color: 'orange' }
    ].map((notebook) => (
      <div key={notebook.num} className={`bg-${notebook.color}-50 border-l-8 border-${notebook.color}-500 p-5 rounded-r-xl shadow-lg hover:shadow-xl transition`}>
        <div className="flex items-start gap-4">
          <div className={`bg-${notebook.color}-500 text-white rounded-full w-16 h-16 flex items-center justify-center flex-shrink-0`}>
            <notebook.icon className="w-8 h-8" />
          </div>
          <div className="flex-1">
            <div className="flex items-baseline gap-3 mb-2">
              <h3 className={`text-2xl font-bold text-${notebook.color}-600`}>Notebook {notebook.num}: {notebook.title}</h3>
              <span className="text-lg text-gray-500 italic">({notebook.subtitle})</span>
            </div>
            <div className="grid grid-cols-3 gap-4 mt-3 text-sm">
              <div>
                <p className="font-semibold text-gray-700 mb-1">What:</p>
                <p className="text-gray-600">{notebook.what}</p>
              </div>
              <div>
                <p className="font-semibold text-gray-700 mb-1">Why:</p>
                <p className="text-gray-600">{notebook.why}</p>
              </div>
              <div>
                <p className="font-semibold text-gray-700 mb-1">Business Value:</p>
                <p className={`text-${notebook.color}-700 font-semibold`}>{notebook.value}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    ))}
  </div>
</div>
);
// SLIDE 6 - Notebook 01: Environment Setup
slides.push(
<div className="h-full flex flex-col p-12 bg-white">
<Header num={6} />
<div className="bg-blue-600 text-white p-4 rounded-t-xl">
<h1 className="text-4xl font-bold">Notebook 01: Data Exploration</h1>
<p className="text-xl mt-2">Cell Analysis: Environment Setup</p>
</div>
  <div className="grid grid-cols-2 gap-6 mt-6">
    <div>
      <div className="bg-gray-900 text-green-400 p-6 rounded-lg font-mono text-sm mb-4">
        <div className="text-gray-400"># Environment Setup</div>
        <div>import pandas as pd</div>
        <div>import numpy as np</div>
        <div>import xgboost</div>
        <div>import torch</div>
        <div>from transformers import AutoModel</div>
        <div className="mt-4 text-white">print("✓ Environment Ready")</div>
        <div className="text-white">print(f"Python: {'{'}sys.version{'}'}")</div>
        <div className="text-white">print(f"PyTorch: {'{'}torch.__version__{'}'}")</div>
      </div>
      
      <div className="bg-green-50 border-2 border-green-600 p-5 rounded-lg">
        <h3 className="text-xl font-bold text-green-700 mb-3 flex items-center gap-2">
          <CheckCircle className="w-6 h-6" />
          What This Cell Does
        </h3>
        <p className="text-gray-700 leading-relaxed">
          Environment setup and validation that imports data science libraries (pandas, numpy, matplotlib, seaborn, scipy), configures visualization defaults, and prints package versions to ensure reproducible fraud detection analysis.
        </p>
      </div>
    </div>

    <div className="space-y-4">
      <div className="bg-purple-50 border-l-4 border-purple-600 p-5 rounded-r-lg">
        <h3 className="text-xl font-bold text-purple-700 mb-3">Business Value</h3>
        <ul className="space-y-2 text-gray-700">
          <li className="flex items-start gap-2">
            <span className="text-purple-600 font-bold">•</span>
            <span><strong>Reproducibility:</strong> Identical results across all analysts</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-purple-600 font-bold">•</span>
            <span><strong>Compliance:</strong> Version tracking satisfies audit requirements</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-purple-600 font-bold">•</span>
            <span><strong>Risk Reduction:</strong> Prevents model failures from version drift</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-purple-600 font-bold">•</span>
            <span><strong>Time Savings:</strong> Cuts onboarding from days to hours</span>
          </li>
        </ul>
      </div>

      <div className="bg-orange-50 border-l-4 border-orange-600 p-5 rounded-r-lg">
        <h3 className="text-xl font-bold text-orange-700 mb-3">Persona Impact</h3>
        <div className="space-y-3 text-sm">
          <div className="bg-white p-3 rounded-lg">
            <p className="font-bold text-gray-800">Data Scientist/ML Engineer</p>
            <p className="text-gray-600">Fast setup with confidence in package stability</p>
          </div>
          <div className="bg-white p-3 rounded-lg">
            <p className="font-bold text-gray-800">IT/Security Leader</p>
            <p className="text-gray-600">CVE-scanned packages with full provenance tracking</p>
          </div>
          <div className="bg-white p-3 rounded-lg">
            <p className="font-bold text-gray-800">Compliance Officer</p>
            <p className="text-gray-600">Automated SBOM documentation for audits</p>
          </div>
        </div>
      </div>

      <div className="bg-green-50 border-2 border-green-600 p-5 rounded-lg">
        <h3 className="text-xl font-bold text-green-700 mb-3">Anaconda Impact</h3>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div className="bg-white p-3 rounded-lg">
            <p className="font-semibold text-gray-700">CVE-Scanned Packages</p>
            <p className="text-green-600 font-bold">vs. Unvetted PyPI</p>
          </div>
          <div className="bg-white p-3 rounded-lg">
            <p className="font-semibold text-gray-700">Reproducible Environments</p>
            <p className="text-green-600 font-bold">Dev = Prod</p>
          </div>
        </div>
        <div className="mt-3 p-3 bg-white rounded-lg text-center">
          <p className="font-bold text-green-700 text-lg">ROI: Prevents $2M-$10M breach</p>
        </div>
      </div>
    </div>
  </div>
</div>
);
// SLIDE 7 - Notebook 01: Class Distribution
slides.push(
<div className="h-full flex flex-col p-12 bg-white">
<Header num={7} />
<div className="bg-blue-600 text-white p-4 rounded-t-xl">
<h1 className="text-4xl font-bold">Notebook 01: Data Exploration</h1>
<p className="text-xl mt-2">Cell Analysis: Class Distribution</p>
</div>
  <div className="grid grid-cols-2 gap-6 mt-6">
    <div>
      <div className="bg-gray-50 border-2 border-gray-300 p-6 rounded-lg mb-4">
        <h3 className="text-2xl font-bold mb-4 text-gray-800">Dataset Statistics</h3>
        <div className="space-y-3">
          <div className="flex justify-between items-center p-3 bg-green-50 rounded-lg">
            <span className="font-semibold text-gray-700">Legitimate Transactions</span>
            <span className="text-2xl font-bold text-green-600">284,315 (99.83%)</span>
          </div>
          <div className="flex justify-between items-center p-3 bg-red-50 rounded-lg">
            <span className="font-semibold text-gray-700">Fraudulent Transactions</span>
            <span className="text-2xl font-bold text-red-600">492 (0.17%)</span>
          </div>
          <div className="flex justify-between items-center p-4 bg-yellow-50 rounded-lg border-2 border-yellow-500">
            <span className="font-bold text-gray-800">Imbalance Ratio</span>
            <span className="text-3xl font-bold text-yellow-700">578:1</span>
          </div>
        </div>
      </div>

      <div className="bg-red-50 border-2 border-red-500 p-5 rounded-lg">
        <h3 className="text-xl font-bold text-red-700 mb-3">⚠️ The "Accuracy Trap"</h3>
        <div className="bg-white p-4 rounded-lg mb-3">
          <p className="text-gray-700 mb-2">A naive model predicting <strong>everything as legitimate:</strong></p>
          <div className="text-3xl font-bold text-red-600">99.83% Accuracy</div>
          <div className="text-xl text-gray-600 mt-1">but catches <strong className="text-red-600">ZERO fraud</strong></div>
        </div>
        <p className="text-sm text-gray-600 italic">This cell exposes that trap before millions are wasted</p>
      </div>
    </div>

    <div className="space-y-4">
      <div className="bg-purple-50 border-l-4 border-purple-600 p-5 rounded-r-lg">
        <h3 className="text-xl font-bold text-purple-700 mb-3">Business Value</h3>
        <ul className="space-y-2 text-gray-700 text-sm">
          <li className="flex items-start gap-2">
            <CheckCircle className="w-5 h-5 text-purple-600 flex-shrink-0 mt-0.5" />
            <span><strong>Prevents "Accuracy Theater":</strong> Naive 99.8% model catches zero fraud—this cell exposes the trap</span>
          </li>
          <li className="flex items-start gap-2">
            <CheckCircle className="w-5 h-5 text-purple-600 flex-shrink-0 mt-0.5" />
            <span><strong>Justifies Technical Investment:</strong> Documents why specialized ML required</span>
          </li>
          <li className="flex items-start gap-2">
            <CheckCircle className="w-5 h-5 text-purple-600 flex-shrink-0 mt-0.5" />
            <span><strong>Operational Planning:</strong> 578:1 ratio determines investigator workload</span>
          </li>
        </ul>
      </div>

      <div className="bg-blue-50 border-l-4 border-blue-600 p-5 rounded-r-lg">
        <h3 className="text-xl font-bold text-blue-700 mb-3">Key Takeaways</h3>
        <ul className="space-y-2 text-gray-700 text-sm">
          <li className="flex items-start gap-2">
            <span className="text-blue-600 font-bold">1.</span>
            <span><strong>Imbalance ratio = operational constraint:</strong> Even 95% recall with 578:1 ratio means 5 false positives per true positive</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-600 font-bold">2.</span>
            <span><strong>Saved PNG = reusable business asset:</strong> Chart becomes standard slide in stakeholder presentations</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-600 font-bold">3.</span>
            <span><strong>Visualization quality signals professionalism:</strong> Enterprise-grade outputs, not academic scratchwork</span>
          </li>
        </ul>
      </div>

      <div className="bg-green-100 border-2 border-green-600 p-4 rounded-lg">
        <div className="text-center">
          <p className="text-sm font-semibold text-gray-700 mb-1">ROI Impact</p>
          <p className="text-2xl font-bold text-green-700">Prevents $500K-$2M</p>
          <p className="text-xs text-gray-600 mt-1">One prevented naive model deployment</p>
        </div>
      </div>
    </div>
  </div>
</div>
);
// SLIDE 8 - Notebook 02: Model Training Architecture
slides.push(
<div className="h-full flex flex-col p-12 bg-white">
<Header num={8} />
<div className="bg-purple-600 text-white p-4 rounded-t-xl">
<h1 className="text-4xl font-bold">Notebook 02: Model Training</h1>
<p className="text-xl mt-2">Hybrid ML+LLM Architecture</p>
</div>
  <div className="mt-6 space-y-6">
    <div className="bg-gradient-to-r from-blue-50 to-blue-100 border-4 border-blue-500 rounded-xl p-6">
      <div className="flex items-center gap-3 mb-4">
        <Zap className="w-10 h-10 text-blue-600" />
        <div>
          <h3 className="text-2xl font-bold text-blue-700">Stage 1: XGBoost (Rapid Screening)</h3>
          <p className="text-gray-600">Processes 10,000 transactions/second</p>
        </div>
      </div>
      <div className="bg-white p-5 rounded-lg">
        <div className="grid grid-cols-3 gap-4 text-center mb-4">
          <div className="bg-blue-50 p-3 rounded-lg">
            <p className="text-sm text-gray-600">Throughput</p>
            <p className="text-2xl font-bold text-blue-600">10K TPS</p>
          </div>
          <div className="bg-blue-50 p-3 rounded-lg">
            <p className="text-sm text-gray-600">Latency</p>
            <p className="text-2xl font-bold text-blue-600">&lt;50ms</p>
          </div>
          <div className="bg-blue-50 p-3 rounded-lg">
            <p className="text-sm text-gray-600">Coverage</p>
            <p className="text-2xl font-bold text-blue-600">95%</p>
          </div>
        </div>
        <div className="text-gray-700 text-sm">
          <strong>Logic:</strong> If XGBoost score &gt; 0.3 (uncertain) → Trigger LLM | Else → Auto-approve (obvious legitimate)
        </div>
      </div>
    </div>

    <div className="bg-gradient-to-r from-purple-50 to-purple-100 border-4 border-purple-500 rounded-xl p-6">
      <div className="flex items-center gap-3 mb-4">
        <Eye className="w-10 h-10 text-purple-600" />
        <div>
          <h3 className="text-2xl font-bold text-purple-700">Stage 2: Qwen 2.5 7B (Deep Analysis)</h3>
          <p className="text-gray-600">Analyzes 5% of high-risk transactions</p>
        </div>
      </div>
      <div className="bg-white p-5 rounded-lg">
        <div className="grid grid-cols-3 gap-4 text-center mb-4">
          <div className="bg-purple-50 p-3 rounded-lg">
            <p className="text-sm text-gray-600">Coverage</p>
            <p className="text-2xl font-bold text-purple-600">5%</p>
          </div>
          <div className="bg-purple-50 p-3 rounded-lg">
            <p className="text-sm text-gray-600">Latency</p>
            <p className="text-2xl font-bold text-purple-600">~2s</p>
          </div>
          <div className="bg-purple-50 p-3 rounded-lg">
            <p className="text-sm text-gray-600">Value</p>
            <p className="text-2xl font-bold text-purple-600">Explainability</p>
          </div>
        </div>
        <div className="bg-purple-50 p-3 rounded-lg text-sm text-gray-700">
          <strong>Output:</strong> "BITCOIN ATM, $2,500, unusual for customer profile → HIGH RISK"
        </div>
      </div>
    </div>

    <div className="bg-green-50 border-2 border-green-600 rounded-xl p-6">
      <h3 className="text-2xl font-bold text-green-700 mb-4 flex items-center gap-2">
        <DollarSign className="w-8 h-8" />
        Selective LLM = Cost Optimization
      </h3>
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-red-50 p-4 rounded-lg border-2 border-red-300">
          <p className="text-sm font-semibold text-gray-700 mb-2">100% LLM Analysis</p>
          <p className="text-3xl font-bold text-red-600">$200K-$1M</p>
          <p className="text-xs text-gray-500 mt-1">monthly compute</p>
        </div>
        <div className="bg-green-100 p-4 rounded-lg border-2 border-green-500">
          <p className="text-sm font-semibold text-gray-700 mb-2">Hybrid (5% LLM)</p>
          <p className="text-3xl font-bold text-green-600">$10K-$50K</p>
          <p className="text-xs text-gray-500 mt-1">monthly compute</p>
        </div>
      </div>
      <div className="mt-4 bg-white p-4 rounded-lg text-center">
        <p className="text-xl font-bold text-green-700">Savings: $150K-$950K monthly</p>
        <p className="text-sm text-gray-600">90-95% compute cost reduction</p>
      </div>
    </div>
  </div>
</div>
);
// SLIDE 9 - Notebook 02: Deployment
slides.push(
<div className="h-full flex flex-col p-12 bg-white">
<Header num={9} />
<h1 className="text-5xl font-bold mb-6 text-gray-900">From Training to Production API in 24 Hours</h1>
  <div className="grid grid-cols-2 gap-6">
    <div className="bg-red-50 border-4 border-red-500 rounded-xl p-6">
      <div className="flex items-center gap-3 mb-4">
        <X className="w-10 h-10 text-red-600" />
        <h3 className="text-2xl font-bold text-red-600">Traditional "Last Mile"</h3>
      </div>
      <div className="space-y-2 text-sm">
        {[
          'Week 1-2: Package as Flask API',
          'Week 3-4: Containerize with Docker',
          'Week 5-6: Authentication, rate limiting',
          'Week 7-8: Configure auto-scaling',
          'Week 9-12: Build monitoring',
          'Week 13-16: Security hardening',
          'Week 17-24: Production deployment'
        ].map((item, i) => (
          <div key={i} className="bg-white p-3 rounded-lg border-l-4 border-red-400">
            <X className="w-4 h-4 text-red-500 inline mr-2" />
            {item}
          </div>
        ))}
      </div>
      <div className="mt-4 bg-red-600 text-white p-4 rounded-lg text-center">
        <p className="text-sm font-semibold">Timeline</p>
        <p className="text-3xl font-bold">24 weeks</p>
        <p className="text-sm mt-2">Cost: $300K-$1M</p>
        <p className="text-sm">Risk: Model outdated by launch</p>
      </div>
    </div>

    <div className="bg-green-50 border-4 border-green-600 rounded-xl p-6">
      <div className="flex items-center gap-3 mb-4">
        <CheckCircle className="w-10 h-10 text-green-600" />
        <h3 className="text-2xl font-bold text-green-600">AI Catalyst Solution</h3>
      </div>
      <div className="space-y-2 text-sm">
        {[
          'Hour 1: Upload notebook + environment.yml',
          'Hour 2-3: Build Docker container',
          'Hour 4-6: Deploy to Kubernetes',
          'Hour 7-8: Configure auto-scaling',
          'Hour 9-24: Testing & validation'
        ].map((item, i) => (
          <div key={i} className="bg-white p-3 rounded-lg border-l-4 border-green-400">
            <CheckCircle className="w-4 h-4 text-green-500 inline mr-2" />
            {item}
          </div>
        ))}
      </div>
      <div className="mt-4 space-y-3">
        <div className="bg-green-600 text-white p-4 rounded-lg text-center">
          <p className="text-sm font-semibold">Timeline</p>
          <p className="text-3xl font-bold">24 hours</p>
          <p className="text-sm mt-2">Cost: $0 additional</p>
          <p className="text-sm">Success: Model relevant at launch</p>
        </div>
        <div className="bg-white border-2 border-green-500 p-4 rounded-lg text-center">
          <p className="text-xl font-bold text-green-700">Savings: $300K-$1M</p>
          <p className="text-sm text-gray-600">23 weeks faster</p>
        </div>
      </div>
    </div>
  </div>
</div>
);
// SLIDE 10 - Notebook 03: ROC Curve
slides.push(
<div className="h-full flex flex-col p-12 bg-white">
<Header num={10} />
<div className="bg-indigo-600 text-white p-4 rounded-t-xl">
<h1 className="text-4xl font-bold">Notebook 03: Advanced Evaluation</h1>
<p className="text-xl mt-2">ROC Curve Analysis & Threshold Optimization</p>
</div>
  <div className="grid grid-cols-2 gap-6 mt-6">
    <div>
      <div className="bg-indigo-50 border-2 border-indigo-500 p-6 rounded-lg mb-4">
        <h3 className="text-2xl font-bold text-indigo-700 mb-4">Performance Metrics</h3>
        <div className="space-y-3">
          {[
            { label: 'ROC-AUC Score', value: '0.9823', percent: '98.23%', color: 'green' },
            { label: 'Optimal Threshold', value: '0.320', percent: 'vs 0.5 default', color: 'blue' },
            { label: 'At Optimal: TPR', value: '98.0%', percent: 'fraud caught', color: 'green' },
            { label: 'At Optimal: FPR', value: '0.3%', percent: 'false alarms', color: 'yellow' }
          ].map((metric, i) => (
            <div key={i} className={`bg-${metric.color}-50 p-3 rounded-lg border-l-4 border-${metric.color}-500`}>
              <div className="flex justify-between items-center">
                <span className="font-semibold text-gray-700">{metric.label}</span>
                <div className="text-right">
                  <span className={`text-2xl font-bold text-${metric.color}-600`}>{metric.value}</span>
                  <p className="text-xs text-gray-500">{metric.percent}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-orange-50 border-2 border-orange-500 p-5 rounded-lg">
        <h3 className="text-xl font-bold text-orange-700 mb-3">💡 Key Insight</h3>
        <p className="text-gray-700 leading-relaxed">
          Optimal threshold is <strong>0.320</strong>, not default <strong>0.5</strong>. Youden's J statistic accounts for class imbalance—common for 0.17% fraud rate datasets.
        </p>
        <div className="mt-3 bg-white p-3 rounded-lg">
          <p className="text-sm font-semibold text-orange-700">Moving from 0.5 → 0.320:</p>
          <p className="text-sm text-gray-600">+2% fraud detection, +0.18% false positives</p>
        </div>
      </div>
    </div>

    <div className="space-y-4">
      <div className="bg-white border-2 border-gray-300 p-6 rounded-lg shadow-lg">
        <h3 className="text-xl font-bold text-gray-800 mb-4 text-center">Threshold Comparison</h3>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b-2 border-gray-300">
              <th className="text-left p-2">Threshold</th>
              <th className="text-right p-2">Recall</th>
              <th className="text-right p-2">FP Rate</th>
              <th className="text-center p-2">Status</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-gray-200 bg-yellow-50">
              <td className="p-2 font-bold">0.320 (Optimal)</td>
              <td className="text-right p-2 font-bold text-green-600">98.0%</td>
              <td className="text-right p-2">0.30%</td>
              <td className="text-center p-2"><Award className="w-5 h-5 text-yellow-500 inline" /></td>
            </tr>
            <tr className="border-b border-gray-200 bg-blue-50">
              <td className="p-2 font-bold">0.500 (Current)</td>
              <td className="text-right p-2 font-bold text-green-600">96.1%</td>
              <td className="text-right p-2">0.12%</td>
              <td className="text-center p-2"><CheckCircle className="w-5 h-5 text-blue-500 inline" /></td>
            </tr>
            <tr className="border-b border-gray-200">
              <td className="p-2">0.700</td>
              <td className="text-right p-2 text-yellow-600">91.8%</td>
              <td className="text-right p-2">0.03%</td>
              <td className="text-center p-2"><AlertCircle className="w-5 h-5 text-gray-400 inline" /></td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="bg-purple-50 border-l-4 border-purple-600 p-5 rounded-r-lg">
        <h3 className="text-xl font-bold text-purple-700 mb-3">Business Value</h3>
        <ul className="space-y-2 text-sm text-gray-700">
          <li className="flex items-start gap-2">
            <CheckCircle className="w-4 h-4 text-purple-600 mt-0.5" />
            <span><strong>Threshold Optimization:</strong> Switching to 0.320 could improve detection 2-5%</span>
          </li>
          <li className="flex items-start gap-2">
            <CheckCircle className="w-4 h-4 text-purple-600 mt-0.5" />
            <span><strong>Cost-Benefit Visualization:</strong> ROC shows FP/TP tradeoff at every threshold</span>
          </li>
          <li className="flex items-start gap-2">
            <CheckCircle className="w-4 h-4 text-purple-600 mt-0.5" />
            <span><strong>Production Tuning:</strong> Can deploy optimal without retraining model</span>
          </li>
        </ul>
      </div>

      <div className="bg-green-100 border-2 border-green-600 p-4 rounded-lg text-center">
        <p className="text-sm font-semibold text-gray-700">ROI Impact</p>
        <p className="text-3xl font-bold text-green-700">$300K-$1.5M</p>
        <p className="text-xs text-gray-600">annually from threshold optimization</p>
      </div>
    </div>
  </div>
</div>
);
// SLIDE 11 - Notebook 04: Financial Impact
slides.push(
<div className="h-full flex flex-col p-12 bg-white">
<Header num={11} />
<div className="bg-green-600 text-white p-4 rounded-t-xl">
<h1 className="text-4xl font-bold">Notebook 04: Business Impact Analysis</h1>
<p className="text-xl mt-2">Translating Technical Excellence to CFO Language</p>
</div>
  <div className="mt-6 space-y-6">
    <div className="grid grid-cols-3 gap-4">
      <div className="bg-gradient-to-br from-green-500 to-green-600 text-white p-6 rounded-xl shadow-lg">
        <DollarSign className="w-12 h-12 mb-3" />
        <p className="text-sm font-semibold mb-1">Fraud Prevention</p>
        <p className="text-4xl font-bold">$673K</p>
        <p className="text-sm mt-1">per year</p>
        <div className="mt-3 pt-3 border-t border-white/30 text-xs">
          374 additional frauds caught monthly
        </div>
      </div>
      
      <div className="bg-gradient-to-br from-blue-500 to-blue-600 text-white p-6 rounded-xl shadow-lg">
        <TrendingUp className="w-12 h-12 mb-3" />
        <p className="text-sm font-semibold mb-1">Operational Savings</p>
        <p className="text-4xl font-bold">$52.7M</p>
        <p className="text-sm mt-1">per year</p>
        <div className="mt-3 pt-3 border-t border-white/30 text-xs">
          58,560 fewer false positives monthly
        </div>
      </div>
      
      <div className="bg-gradient-to-br from-purple-500 to-purple-600 text-white p-6 rounded-xl shadow-lg">
        <Award className="w-12 h-12 mb-3" />
        <p className="text-sm font-semibold mb-1">Total Annual Value</p>
        <p className="text-4xl font-bold">$53.4M</p>
        <p className="text-sm mt-1">per year</p>
        <div className="mt-3 pt-3 border-t border-white/30 text-xs">
          Combined fraud + efficiency
        </div>
      </div>
    </div>

    <div className="bg-gradient-to-r from-gray-50 to-gray-100 border-2 border-gray-400 rounded-xl p-6">
      <h3 className="text-2xl font-bold text-gray-800 mb-4 text-center">ROI Calculation</h3>
      <div className="grid grid-cols-2 gap-6">
        <div>
          <p className="text-lg font-semibold text-gray-700 mb-3">First-Year Costs</p>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between p-2 bg-white rounded">
              <span>Platform License</span>
              <span className="font-bold">$50K-$150K</span>
            </div>
            <div className="flex justify-between p-2 bg-white rounded">
              <span>AI Catalyst</span>
              <span className="font-bold">$30K-$80K</span>
            </div>
            <div className="flex justify-between p-2 bg-white rounded">
              <span>Implementation</span>
              <span className="font-bold">$20K-$70K</span>
            </div>
            <div className="flex justify-between p-3 bg-red-100 rounded border-2 border-red-300 font-bold">
              <span>Total Investment</span>
              <span className="text-red-600">$100K-$300K</span>
            </div>
          </div>
        </div>
        <div>
          <p className="text-lg font-semibold text-gray-700 mb-3">Returns</p>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between p-2 bg-white rounded">
              <span>Annual Value</span>
              <span className="font-bold text-green-600">$53.4M</span>
            </div>
            <div className="flex justify-between p-2 bg-white rounded">
              <span>ROI</span>
              <span className="font-bold text-green-600">17,792%</span>
            </div>
            <div className="flex justify-between p-2 bg-white rounded">
              <span>Payback Period</span>
              <span className="font-bold text-green-600">2-7 days</span>
            </div>
            <div className="flex justify-between p-3 bg-green-100 rounded border-2 border-green-500 font-bold">
              <span>Net Benefit (Yr 1)</span>
              <span className="text-green-600">$53.1M-$53.3M</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div className="bg-blue-50 border-l-4 border-blue-600 p-5 rounded-r-lg">
      <h3 className="text-xl font-bold text-blue-700 mb-3">CFO-Ready Business Case</h3>
      <div className="text-gray-700 space-y-2 text-sm leading-relaxed">
        <p>• <strong>$53.4M annual value</strong> vs. <strong>$110K-$330K total cost</strong></p>
        <p>• <strong>Dual value streams:</strong> Fraud prevention ($594K) + operational efficiency ($724K) = $1.3M monthly</p>
        <p>• <strong>Payback validation:</strong> 2-7 days (model pays for itself in first week)</p>
        <p>• <strong>Total Cost of Ownership:</strong> Fraud losses + customer friction + review costs transparently modeled</p>
      </div>
    </div>
  </div>
</div>
);
// SLIDE 12 - Notebook 05: Interactive Demo
slides.push(
<div className="h-full flex flex-col p-12 bg-white">
<Header num={12} />
<div className="bg-orange-600 text-white p-4 rounded-t-xl">
<h1 className="text-4xl font-bold">Notebook 05: Interactive Demo</h1>
<p className="text-xl mt-2">Hands-On Validation = 40-60% Higher Close Rates</p>
</div>
  <div className="mt-6 grid grid-cols-3 gap-4">
    <div className="bg-gradient-to-br from-blue-50 to-blue-100 border-3 border-blue-500 rounded-xl p-5">
      <div className="flex items-center gap-2 mb-3">
        <Terminal className="w-8 h-8 text-blue-600" />
        <h3 className="text-xl font-bold text-blue-700">Manual Testing</h3>
      </div>
      <div className="bg-white p-4 rounded-lg mb-3 shadow">
        <div className="space-y-2 text-sm">
          <div>
            <label className="text-gray-600 text-xs">Merchant:</label>
            <input type="text" value="BITCOIN ATM" readOnly className="w-full p-2 border rounded mt-1 bg-gray-50" />
          </div>
          <div>
            <label className="text-gray-600 text-xs">Amount:</label>
            <input type="text" value="$2,500.00" readOnly className="w-full p-2 border rounded mt-1 bg-gray-50" />
          </div>
          <button className="w-full bg-blue-600 text-white py-2 rounded font-semibold hover:bg-blue-700">
            Analyze Transaction
          </button>
        </div>
      </div>
      <div className="bg-red-600 text-white p-3 rounded-lg text-center">
        <p className="font-bold">🔴 BLOCK TRANSACTION</p>
        <p className="text-xs mt-1">HIGH RISK</p>
      </div>
      <div className="mt-2 text-xs text-gray-600 bg-white p-2 rounded">
        <strong>Explanation:</strong> Suspicious keywords, high amount, matches fraud patterns
      </div>
    </div>

    <div className="bg-gradient-to-br from-purple-50 to-purple-100 border-3 border-purple-500 rounded-xl p-5">
      <div className="flex items-center gap-2 mb-3">
        <Target className="w-8 h-8 text-purple-600" />
        <h3 className="text-xl font-bold text-purple-700">Pre-Built Scenarios</h3>
      </div>
      <div className="space-y-2">
        {[
          { name: 'Normal Grocery', decision: 'APPROVE', color: 'green' },
          { name: 'Bitcoin ATM', decision: 'BLOCK', color: 'red' },
          { name: 'Business Lunch', decision: 'APPROVE', color: 'green' },
          { name: 'Large Wire Transfer', decision: 'REVIEW', color: 'yellow' },
          { name: 'Streaming Service', decision: 'APPROVE', color: 'green' },
          { name: 'Online Casino', decision: 'BLOCK', color: 'red' }
        ].map((scenario, i) => (
          <button key={i} className={`w-full bg-${scenario.color}-500 hover:bg-${scenario.color}-600 text-white p-2 rounded text-sm font-semibold transition`}>
            {scenario.name} → {scenario.decision}
          </button>
        ))}
      </div>
    </div>

    <div className="bg-gradient-to-br from-green-50 to-green-100 border-3 border-green-500 rounded-xl p-5">
      <div className="flex items-center gap-2 mb-3">
        <Server className="w-8 h-8 text-green-600" />
        <h3 className="text-xl font-bold text-green-700">API Testing</h3>
      </div>
      <div className="bg-white p-4 rounded-lg mb-3 shadow">
        <button className="w-full bg-green-600 text-white py-2 rounded font-semibold mb-3 hover:bg-green-700">
          Test API Connection
        </button>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between p-2 bg-green-50 rounded">
            <span>Anaconda Connect</span>
            <span className="text-green-600 font-bold">✓ Online</span>
          </div>
          <div className="flex justify-between p-2 bg-green-50 rounded">
            <span>AI Navigator</span>
            <span className="text-green-600 font-bold">✓ Online</span>
          </div>
        </div>
      </div>
      <div className="bg-white p-3 rounded-lg border-2 border-green-300">
        <p className="text-xs font-semibold text-gray-700 mb-2">API Response:</p>
        <div className="space-y-1 text-xs text-gray-600">
          <p>• Source: Anaconda Connect</p>
          <p>• Latency: 145ms</p>
          <p>• Decision: <strong className="text-green-600">APPROVE</strong></p>
        </div>
      </div>
      <div className="mt-2 text-xs text-gray-600 italic text-center">
        This IS production, not a simulation
      </div>
    </div>
  </div>

  <div className="mt-4 bg-yellow-50 border-2 border-yellow-500 p-4 rounded-lg">
    <p className="text-center text-lg font-bold text-yellow-800">
      💡 Value: Prospects who test fraud detection themselves have 40-60% higher close rates
    </p>
  </div>
</div>
);
// SLIDE 13 - Partner Overview
slides.push(
<div className="h-full flex flex-col p-12 bg-white">
<Header num={13} />
<h1 className="text-5xl font-bold mb-6 text-gray-900">Partner Integration Overview</h1>
<h2 className="text-3xl text-gray-600 mb-8">Anaconda = Infrastructure Foundation Enabling Best-in-Class Partners</h2>
  <div className="space-y-6">
    <div className="bg-green-600 text-white p-6 rounded-xl shadow-lg">
      <div className="flex items-center gap-3 mb-3">
        <Package className="w-10 h-10" />
        <h3 className="text-2xl font-bold">ANACONDA SELF-HOSTED CORE (Foundation Layer)</h3>
      </div>
      <div className="grid grid-cols-4 gap-3 text-sm">
        {[
          'CVE-scanned Python + 9,000 packages',
          '$10M legal indemnification',
          'Reproducible environments (dev = prod)',
          'Air-gapped deployment for compliance'
        ].map((item, i) => (
          <div key={i} className="bg-white/10 p-3 rounded-lg backdrop-blur">
            <CheckCircle className="w-5 h-5 mb-1" />
            <p>{item}</p>
          </div>
        ))}
      </div>
    </div>

    <div className="flex justify-center">
      <div className="text-2xl font-bold text-gray-400">enables ↓</div>
    </div>

    <div className="grid grid-cols-3 gap-4">
      <div className="bg-blue-50 border-2 border-blue-500 rounded-xl p-5">
        <Database className="w-10 h-10 text-blue-600 mb-3" />
        <h3 className="text-xl font-bold text-blue-700 mb-3">Data Layer</h3>
        <div className="space-y-2 text-sm">
          <div className="bg-white p-3 rounded-lg border-l-4 border-blue-400 font-semibold">
            ❄️ Snowflake
          </div>
          <div className="bg-white p-3 rounded-lg border-l-4 border-blue-400 font-semibold">
            🧱 Databricks
          </div>
        </div>
      </div>

      <div className="bg-purple-50 border-2 border-purple-500 rounded-xl p-5">
        <Cpu className="w-10 h-10 text-purple-600 mb-3" />
        <h3 className="text-xl font-bold text-purple-700 mb-3">Compute Layer</h3>
        <div className="space-y-2 text-sm">
          <div className="bg-white p-3 rounded-lg border-l-4 border-purple-400 font-semibold">
            ☁️ AWS / Azure / GCP
          </div>
          <div className="bg-white p-3 rounded-lg border-l-4 border-purple-400 font-semibold">
            🟢 NVIDIA GPUs
          </div>
          <div className="bg-white p-3 rounded-lg border-l-4 border-purple-400 font-semibold">
            🔥 PyTorch
          </div>
        </div>
      </div>

      <div className="bg-orange-50 border-2 border-orange-500 rounded-xl p-5">
        <Box className="w-10 h-10 text-orange-600 mb-3" />
        <h3 className="text-xl font-bold text-orange-700 mb-3">Deploy Layer</h3>
        <div className="space-y-2 text-sm">
          <div className="bg-white p-3 rounded-lg border-l-4 border-orange-400 font-semibold">
            🐳 Docker
          </div>
          <div className="bg-white p-3 rounded-lg border-l-4 border-orange-400 font-semibold">
            ⚙️ Kubernetes
          </div>
        </div>
      </div>
    </div>

    <div className="bg-green-50 border-2 border-green-600 p-5 rounded-lg">
      <p className="text-xl font-bold text-green-700 text-center mb-3">🎯 Core Positioning</p>
      <p className="text-gray-700 text-center text-lg leading-relaxed">
        "Anaconda doesn't compete with these partners—we make them better. Snowflake stores your data, <strong>we secure the pipeline</strong>. Docker containerizes your ML, <strong>we harden the images</strong>. AWS provides infrastructure, <strong>we provide the ML platform</strong>."
      </p>
    </div>
  </div>
</div>
);
// SLIDE 14 - Docker Integration
slides.push(
<div className="h-full flex flex-col p-12 bg-white">
<Header num={14} />
<h1 className="text-5xl font-bold mb-6 text-gray-900">🐳 Docker Integration</h1>
<h2 className="text-3xl text-gray-600 mb-8">Hardened Container Images = Security by Default (Q2 2026)</h2>
  <div className="grid grid-cols-2 gap-6">
    <div className="bg-red-50 border-4 border-red-500 rounded-xl p-6">
      <div className="flex items-center gap-3 mb-4">
        <X className="w-10 h-10 text-red-600" />
        <h3 className="text-2xl font-bold text-red-600">Standard Container</h3>
      </div>
      <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-xs mb-4">
        <div className="text-gray-400"># Standard fraud detection</div>
        <div>FROM python:3.11-slim</div>
        <div>RUN pip install torch xgboost</div>
        <div className="text-gray-400 mt-2"># ❌ Unscanned base image</div>
        <div className="text-gray-400"># ❌ PyPI (no CVE scanning)</div>
        <div className="text-gray-400"># ❌ No legal protection</div>
      </div>
      <div className="space-y-2">
        {[
          { risk: 'Unknown vulnerabilities', impact: 'Breach risk' },
          { risk: 'No legal protection', impact: '$2M-$10M liability' },
          { risk: 'Version drift', impact: 'Unpredictable builds' }
        ].map((item, i) => (
          <div key={i} className="bg-white p-3 rounded-lg border-l-4 border-red-400">
            <p className="text-sm"><strong className="text-red-600">Risk:</strong> {item.risk}</p>
            <p className="text-xs text-gray-600">Impact: {item.impact}</p>
          </div>
        ))}
      </div>
    </div>

    <div className="bg-green-50 border-4 border-green-600 rounded-xl p-6">
      <div className="flex items-center gap-3 mb-4">
        <CheckCircle className="w-10 h-10 text-green-600" />
        <h3 className="text-2xl font-bold text-green-600">Anaconda Hardened</h3>
      </div>
      <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-xs mb-4">
        <div className="text-gray-400"># Anaconda hardened (Q2 2026)</div>
        <div>FROM anaconda/hardened-python:</div>
        <div>     3.11-cuda12.1</div>
        <div className="text-green-400 mt-2"># ✓ CVE-scanned base + CUDA</div>
        <div className="text-green-400"># ✓ 500+ deps verified</div>
        <div className="text-green-400"># ✓ $10M indemnification</div>
      </div>
      <div className="space-y-2">
        {[
          { benefit: 'Every layer CVE-scanned', value: 'Monthly updates' },
          { benefit: 'Pre-configured CUDA', value: 'GPU-ready instantly' },
          { benefit: 'Reproducible builds', value: 'Same image 6mo later' },
          { benefit: 'Legal protection', value: '$10M coverage' }
        ].map((item, i) => (
          <div key={i} className="bg-white p-3 rounded-lg border-l-4 border-green-400">
            <p className="text-sm"><strong className="text-green-600">✓</strong> {item.benefit}</p>
            <p className="text-xs text-gray-600">{item.value}</p>
          </div>
        ))}
      </div>
    </div>
  </div>

  <div className="mt-4 grid grid-cols-3 gap-4">
    <div className="bg-gray-100 p-4 rounded-lg text-center">
      <p className="text-sm text-gray-600">Time Saved</p>
      <p className="text-2xl font-bold text-green-600">2-4 weeks</p>
    </div>
    <div className="bg-gray-100 p-4 rounded-lg text-center">
      <p className="text-sm text-gray-600">Cost Saved</p>
      <p className="text-2xl font-bold text-green-600">$50K-$100K</p>
    </div>
    <div className="bg-gray-100 p-4 rounded-lg text-center">
      <p className="text-sm text-gray-600">Risk Avoided</p>
      <p className="text-2xl font-bold text-green-600">$2M-$10M</p>
    </div>
  </div>
</div>
);
// SLIDE 15 - Snowflake Integration
slides.push(
<div className="h-full flex flex-col p-12 bg-white">
<Header num={15} />
<h1 className="text-5xl font-bold mb-6 text-gray-900">❄️ Snowflake Integration</h1>
<h2 className="text-3xl text-gray-600 mb-8">In-Database Fraud Scoring (Q2 2026 Snowpark)</h2>
  <div className="grid grid-cols-2 gap-6">
    <div className="bg-red-50 border-4 border-red-500 rounded-xl p-6">
      <h3 className="text-2xl font-bold text-red-600 mb-4">Traditional Architecture</h3>
      <div className="bg-white p-5 rounded-lg">
        <div className="flex flex-col items-center gap-4">
          <div className="w-full bg-blue-100 border-2 border-blue-500 p-4 rounded-lg text-center">
            <Database className="w-8 h-8 mx-auto mb-2 text-blue-600" />
            <p className="font-bold">Snowflake</p>
            <p className="text-sm text-gray-600">2.3B transactions</p>
          </div>
          <div className="flex items-center gap-2">
            <div className="text-red-600 font-mono text-sm">→ API Call →</div>
            <div className="bg-red-100 px-3 py-1 rounded text-sm font-bold text-red-600">50-200ms</div>
          </div>
          <div className="w-full bg-purple-100 border-2 border-purple-500 p-4 rounded-lg text-center">
            <Server className="w-8 h-8 mx-auto mb-2 text-purple-600" />
            <p className="font-bold">Fraud Detection API</p>
            <p className="text-sm text-gray-600">AWS/Azure</p>
          </div>
        </div>
        <div className="mt-4 p-3 bg-red-100 rounded-lg text-center">
          <p className="text-sm font-semibold text-gray-700">Total Latency</p>
          <p className="text-2xl font-bold text-red-600">100-400ms</p>
          <p className="text-xs text-gray-600">Network overhead</p>
        </div>
      </div>
    </div>

    <div className="bg-green-50 border-4 border-green-600 rounded-xl p-6">
      <h3 className="text-2xl font-bold text-green-600 mb-4">Q2 2026: Snowpark Integration</h3>
      <div className="bg-white p-5 rounded-lg">
        <div className="bg-blue-100 border-2 border-blue-500 p-4 rounded-lg text-center mb-4">
          <Database className="w-12 h-12 mx-auto mb-2 text-blue-600" />
          <p className="font-bold text-lg">Snowflake</p>
          <div className="mt-3 bg-green-100 p-3 rounded-lg border-2 border-green-500">
            <p className="text-sm font-semibold text-gray-700">Fraud Detection INSIDE Snowflake</p>
            <p className="text-xs text-gray-600 mt-1">Snowpark UDF powered by Anaconda</p>
          </div>
        </div>
        
        <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-xs mb-4">
          <div className="text-gray-400"># In-database fraud scoring</div>
          <div>@snowpark_udf</div>
          <div>def score_fraud(merchant, amt):</div>
          <div className="ml-4">return model.predict(...)</div>
          <div className="text-gray-400 mt-2"># Zero data movement</div>
        </div>
        
        <div className="p-3 bg-green-100 rounded-lg text-center">
          <p className="text-sm font-semibold text-gray-700">Total Latency</p>
          <p className="text-3xl font-bold text-green-600">&lt;5ms</p>
          <p className="text-xs text-gray-600">In-database execution</p>
        </div>
      </div>
    </div>
  </div>

  <div className="mt-6 grid grid-cols-3 gap-4">
    <div className="bg-blue-50 p-4 rounded-lg text-center border-2 border-blue-300">
      <p className="text-sm text-gray-600 mb-1">Latency Improvement</p>
      <p className="text-3xl font-bold text-blue-600">10-40x faster</p>
    </div>
    <div className="bg-green-50 p-4 rounded-lg text-center border-2 border-green-300">
      <p className="text-sm text-gray-600 mb-1">Data Movement</p>
      <p className="text-3xl font-bold text-green-600">Zero</p>
    </div>
    <div className="bg-purple-50 p-4 rounded-lg text-center border-2 border-purple-300">
      <p className="text-sm text-gray-600 mb-1">ROI Impact</p>
      <p className="text-2xl font-bold text-purple-600">$5M-$20M</p>
    </div>
  </div>
</div>
);
// SLIDE 16 - AWS Multi-Cloud
slides.push(
<div className="h-full flex flex-col p-12 bg-white">
<Header num={16} />
<h1 className="text-5xl font-bold mb-6 text-gray-900">☁️ Multi-Cloud Deployment</h1>
<h2 className="text-3xl text-gray-600 mb-8">Deploy Anywhere = Zero Vendor Lock-In</h2>
  <div className="mb-6 bg-blue-50 border-2 border-blue-500 p-6 rounded-xl">
    <h3 className="text-2xl font-bold text-blue-700 mb-4 text-center">One Development Environment, Any Cloud</h3>
    <div className="flex items-center justify-center gap-4">
      <div className="bg-white p-4 rounded-lg shadow border-2 border-blue-300">
        <FileCode className="w-10 h-10 text-blue-600 mx-auto mb-2" />
        <p className="text-center font-bold text-gray-800">Develop in Anaconda</p>
        <p className="text-sm text-center text-gray-600 mt-1">Notebooks + Self-Hosted Core</p>
      </div>
      <ArrowRight className="w-8 h-8 text-gray-400" />
      <div className="text-lg font-bold text-gray-600">Same Code</div>
      <ArrowRight className="w-8 h-8 text-gray-400" />
      <div className="flex gap-3">
        {['AWS', 'Azure', 'GCP', 'On-Prem'].map((cloud, i) => (
          <div key={i} className="bg-white p-3 rounded-lg shadow border-2 border-green-500">
            <Cloud className="w-8 h-8 text-green-600 mx-auto mb-1" />
            <p className="text-center font-bold text-sm">{cloud}</p>
          </div>
        ))}
      </div>
    </div>
  </div>

  <div className="grid grid-cols-3 gap-4 mb-6">
    {[
      { cloud: 'AWS', status: 'Available Now', time: '30-60 min', color: 'orange' },
      { cloud: 'Azure', status: 'Q1 2026', time: '30-60 min', color: 'blue' },
      { cloud: 'GCP', status: 'Q2 2026', time: '30-60 min', color: 'green' }
    ].map((item, i) => (
      <div key={i} className={`bg-${item.color}-50 border-2 border-${item.color}-500 p-5 rounded-lg`}>
        <h3 className={`text-xl font-bold text-${item.color}-700 mb-3`}>{item.cloud}</h3>
        <div className="space-y-2 text-sm">
          <div className="bg-white p-2 rounded">
            <span className="text-gray-600">Status:</span>
            <span className={`ml-2 font-bold text-${item.color}-600`}>{item.status}</span>
          </div>
          <div className="bg-white p-2 rounded">
            <span className="text-gray-600">Deploy Time:</span>
            <span className={`ml-2 font-bold text-${item.color}-600`}>{item.time}</span>
          </div>
          <div className="bg-white p-2 rounded">
            <span className="text-gray-600">Billing:</span>
            <span className="ml-2 font-bold text-gray-700">Marketplace</span>
          </div>
        </div>
      </div>
    ))}
  </div>

  <div className="bg-purple-50 border-l-4 border-purple-600 p-6 rounded-r-lg">
    <h3 className="text-2xl font-bold text-purple-700 mb-4">Real-World Multi-Cloud Scenario: Global Bank</h3>
    <div className="grid grid-cols-2 gap-4">
      {[
        { region: 'US Transactions', cloud: 'AWS US-East-1', reason: 'FINRA compliance' },
        { region: 'EU Transactions', cloud: 'Azure EU-West', reason: 'GDPR compliance' },
        { region: 'APAC Transactions', cloud: 'GCP Asia-Southeast', reason: 'Data residency' },
        { region: 'China Transactions', cloud: 'On-Premises', reason: 'Government requirements' }
      ].map((deployment, i) => (
        <div key={i} className="bg-white p-4 rounded-lg border-2 border-purple-300">
          <p className="font-bold text-gray-800">{deployment.region}</p>
          <p className="text-purple-600 font-semibold mt-1">{deployment.cloud}</p>
          <p className="text-xs text-gray-500 mt-1 italic">{deployment.reason}</p>
        </div>
      ))}
    </div>
    <div className="mt-4 bg-white p-4 rounded-lg border-2 border-purple-500 text-center">
      <p className="font-bold text-purple-700 text-lg">One Fraud Model, Four Deployments</p>
      <p className="text-sm text-gray-600 mt-1">Same Anaconda Platform, Different Infrastructure</p>
    </div>
  </div>

  <div className="mt-4 grid grid-cols-3 gap-3">
    <div className="bg-green-100 p-3 rounded-lg text-center">
      <p className="text-xs text-gray-600">Cloud Negotiation</p>
      <p className="text-xl font-bold text-green-600">20-30% discount</p>
    </div>
    <div className="bg-green-100 p-3 rounded-lg text-center">
      <p className="text-xs text-gray-600">Procurement Speed</p>
      <p className="text-xl font-bold text-green-600">6mo → 1 day</p>
    </div>
    <div className="bg-green-100 p-3 rounded-lg text-center">
      <p className="text-xs text-gray-600">Migration Insurance</p>
      <p className="text-xl font-bold text-green-600">$500K-$2M</p>
    </div>
  </div>
</div>
);
// SLIDE 17 - NVIDIA Integration
slides.push(
<div className="h-full flex flex-col p-12 bg-white">
<Header num={17} />
<h1 className="text-5xl font-bold mb-6 text-gray-900">🟢 NVIDIA Integration</h1>
<h2 className="text-3xl text-gray-600 mb-8">GPU-Accelerated Fraud Detection = 10x Faster, 40% Cheaper</h2>
  <div className="grid grid-cols-2 gap-6 mb-6">
    <div className="bg-gray-50 border-4 border-gray-400 rounded-xl p-6">
      <h3 className="text-2xl font-bold text-gray-700 mb-4 text-center">CPU Performance</h3>
      <div className="space-y-3">
        <div className="bg-white p-4 rounded-lg border-l-4 border-gray-500">
          <div className="flex justify-between items-center">
            <span className="text-gray-700">XGBoost Inference</span>
            <span className="text-2xl font-bold text-gray-600">50ms</span>
          </div>
        </div>
        <div className="bg-white p-4 rounded-lg border-l-4 border-gray-500">
          <div className="flex justify-between items-center">
            <span className="text-gray-700">LLM Analysis (Qwen 7B)</span>
            <span className="text-2xl font-bold text-gray-600">2,000ms</span>
          </div>
        </div>
        <div className="bg-red-100 p-4 rounded-lg border-2 border-red-400">
          <div className="flex justify-between items-center">
            <span className="font-bold text-gray-800">Total Hybrid Latency</span>
            <span className="text-3xl font-bold text-red-600">2,050ms</span>
          </div>
        </div>
        <div className="bg-white p-4 rounded-lg border-l-4 border-gray-500">
          <div className="flex justify-between items-center">
            <span className="text-gray-700">Throughput</span>
            <span className="text-2xl font-bold text-gray-600">500 TPS</span>
          </div>
          <p className="text-xs text-gray-500 mt-1">10 CPU instances needed</p>
        </div>
      </div>
    </div>

    <div className="bg-green-50 border-4 border-green-600 rounded-xl p-6">
      <h3 className="text-2xl font-bold text-green-700 mb-4 text-center flex items-center justify-center gap-2">
        <Zap className="w-8 h-8" />
        GPU Performance (NVIDIA A10G)
      </h3>
      <div className="space-y-3">
        <div className="bg-white p-4 rounded-lg border-l-4 border-green-500">
          <div className="flex justify-between items-center">
            <span className="text-gray-700">XGBoost Inference</span>
            <div className="text-right">
              <span className="text-2xl font-bold text-green-600">10ms</span>
              <p className="text-xs text-green-600">5x faster ⚡</p>
            </div>
          </div>
        </div>
        <div className="bg-white p-4 rounded-lg border-l-4 border-green-500">
          <div className="flex justify-between items-center">
            <span className="text-gray-700">LLM Analysis (Qwen 7B)</span>
            <div className="text-right">
              <span className="text-2xl font-bold text-green-600">200ms</span>
              <p className="text-xs text-green-600">10x faster ⚡</p>
            </div>
          </div>
        </div>
        <div className="bg-green-100 p-4 rounded-lg border-2 border-green-500">
          <div className="flex justify-between items-center">
            <span className="font-bold text-gray-800">Total Hybrid Latency</span>
            <div className="text-right">
              <span className="text-3xl font-bold text-green-600">210ms</span>
              <p className="text-xs text-green-600 font-bold">10x faster ⚡⚡⚡</p>
            </div>
          </div>
        </div>
        <div className="bg-white p-4 rounded-lg border-l-4 border-green-500">
          <div className="flex justify-between items-center">
            <span className="text-gray-700">Throughput</span>
            <div className="text-right">
              <span className="text-2xl font-bold text-green-600">5,000 TPS</span>
              <p className="text-xs text-green-600">10x more ⚡</p>
            </div>
          </div>
          <p className="text-xs text-gray-500 mt-1">Single GPU instance</p>
        </div>
      </div>
    </div>
  </div>

  <div className="grid grid-cols-2 gap-6">
    <div className="bg-red-50 p-5 rounded-lg border-2 border-red-400">
      <h3 className="text-xl font-bold text-red-600 mb-3 text-center">CPU-Only Infrastructure</h3>
      <div className="space-y-2 text-sm">
        <div className="bg-white p-2 rounded flex justify-between">
          <span>10 × c5.4xlarge ($0.68/hr)</span>
          <span className="font-bold">$6.80/hr</span>
        </div>
        <div className="bg-white p-2 rounded flex justify-between">
          <span>Monthly Cost</span>
          <span className="font-bold">$4,896</span>
        </div>
        <div className="bg-white p-2 rounded flex justify-between">
          <span>Per Million Transactions</span>
          <span className="font-bold">$2.72</span>
        </div>
      </div>
    </div>

    <div className="bg-green-50 p-5 rounded-lg border-2 border-green-500">
      <h3 className="text-xl font-bold text-green-600 mb-3 text-center">GPU-Accelerated</h3>
      <div className="space-y-2 text-sm">
        <div className="bg-white p-2 rounded flex justify-between">
          <span>1 × g5.2xlarge ($1.21/hr)</span>
          <span className="font-bold">$1.21/hr</span>
        </div>
        <div className="bg-white p-2 rounded flex justify-between">
          <span>Monthly Cost</span>
          <span className="font-bold text-green-600">$871</span>
        </div>
        <div className="bg-white p-2 rounded flex justify-between">
          <span>Per Million Transactions</span>
          <span className="font-bold text-green-600">$0.05</span>
        </div>
      </div>
    </div>
  </div>

  <div className="mt-4 bg-gradient-to-r from-green-500 to-green-600 text-white p-6 rounded-xl text-center shadow-lg">
    <p className="text-2xl font-bold mb-2">Annual Savings: $48,300/year</p>
    <p className="text-xl">82% Infrastructure Cost Reduction</p>
    <p className="text-sm mt-2 opacity-90">Q2 2026: NVIDIA Partnership delivers optimized CUDA packages</p>
  </div>
</div>
);
// SLIDE 18 - 2026 Roadmap Overview
slides.push(
<div className="h-full flex flex-col p-12 bg-white">
<Header num={18} />
<h1 className="text-5xl font-bold mb-6 text-gray-900">2026 Strategic Vision</h1>
<h2 className="text-3xl text-gray-600 mb-8">Solving the "Last Mile Problem"</h2>
  <div className="grid grid-cols-3 gap-6">
    {[
      {
        num: '1',
        icon: Users,
        title: 'Premier Platform for Open Source AI Collaboration',
        desc: 'Establish Anaconda as the undisputed choice where data scientists, engineers, and business teams unite to create transformative AI',
        fraud: 'Data scientists develop in Notebooks → Engineers deploy via AI Catalyst → Business teams monitor dashboards',
        color: 'green'
      },
      {
        num: '2',
        icon: GitBranch,
        title: 'Bridge from Experimentation to Production',
        desc: 'Eliminate the gap between prototype and production with seamless workflows',
        fraud: 'Notebook → AI Catalyst → Production API in 24 hours. Same environment dev/staging/prod.',
        color: 'blue'
      },
      {
        num: '3',
        icon: Shield,
        title: 'World-Class Developer Experience with Enterprise Security',
        desc: 'Empower developers with tools they love while ensuring enterprise-grade security',
        fraud: 'Data scientists use Jupyter/conda → Security gets CVE scanning + $10M indemnification automatically',
        color: 'purple'
      }
    ].map((pillar) => (
      <div key={pillar.num} className={`bg-${pillar.color}-50 border-4 border-${pillar.color}-500 rounded-xl p-6`}>
        <div className="flex items-center justify-center mb-4">
          <div className={`bg-${pillar.color}-600 text-white rounded-full w-16 h-16 flex items-center justify-center text-3xl font-bold`}>
            {pillar.num}
          </div>
        </div>
        <pillar.icon className={`w-12 h-12 text-${pillar.color}-600 mx-auto mb-4`} />
        <h3 className={`text-xl font-bold text-${pillar.color}-700 mb-3 text-center leading-tight`}>{pillar.title}</h3>
        <p className="text-sm text-gray-600 mb-4 leading-relaxed italic">"{pillar.desc}"</p>
        <div className={`bg-white p-3 rounded-lg border-2 border-${pillar.color}-300`}>
          <p className="text-xs font-semibold text-gray-700 mb-1">Fraud Detection Translation:</p>
          <p className="text-xs text-gray-600 leading-relaxed">{pillar.fraud}</p>
        </div>
      </div>
    ))}
  </div>

  <div className="mt-6 bg-yellow-50 border-2 border-yellow-500 p-5 rounded-lg">
    <p className="text-center text-xl font-bold text-yellow-800">
      💡 These three pillars converge in 2026: Explainable AI (regulation) + pip/uv support (developer reality) + Multi-cloud (business strategy)
    </p>
  </div>
</div>
);
// SLIDE 19 - 2026 Game Changers Q1
slides.push(
<div className="h-full flex flex-col p-12 bg-white">
<Header num={19} />
<h1 className="text-5xl font-bold mb-6 text-gray-900">2026 Game-Changing Enhancements</h1>
<h2 className="text-3xl text-gray-600 mb-8">Q1 2026: Market Expansion (2-3x Addressable Market)</h2>
  <div className="space-y-5">
    <div className="bg-gradient-to-r from-blue-500 to-blue-600 text-white p-6 rounded-xl shadow-lg">
      <div className="flex items-center gap-3 mb-3">
        <Package className="w-10 h-10" />
        <h3 className="text-3xl font-bold">1️⃣ pip/uv Wheels Support</h3>
      </div>
      <div className="grid grid-cols-2 gap-6 mt-4">
        <div className="bg-white/10 backdrop-blur p-4 rounded-lg">
          <p className="text-sm mb-2 font-semibold">Before: Only conda users</p>
          <div className="bg-gray-900 text-green-400 font-mono text-xs p-3 rounded">
            conda install pytorch xgboost
          </div>
          <p className="text-xs mt-2 opacity-90">~10M developers</p>
        </div>
        <div className="bg-white/10 backdrop-blur p-4 rounded-lg">
          <p className="text-sm mb-2 font-semibold">After Q1 2026: pip/uv users</p>
          <div className="bg-gray-900 text-green-400 font-mono text-xs p-3 rounded">
            pip install --index-url<br/>
            https://repo.anaconda.cloud<br/>
            pytorch
          </div>
          <p className="text-xs mt-2 opacity-90">+15M developers (2.5x market)</p>
        </div>
      </div>
      <div className="mt-4 bg-white p-4 rounded-lg">
        <div className="grid grid-cols-3 gap-3 text-center text-gray-800">
          <div>
            <CheckCircle className="w-6 h-6 mx-auto text-blue-600 mb-1" />
            <p className="text-xs font-semibold">Same CVE scanning</p>
          </div>
          <div>
            <CheckCircle className="w-6 h-6 mx-auto text-blue-600 mb-1" />
            <p className="text-xs font-semibold">Same $10M indemnification</p>
          </div>
          <div>
            <CheckCircle className="w-6 h-6 mx-auto text-blue-600 mb-1" />
            <p className="text-xs font-semibold">Zero workflow change</p>
          </div>
        </div>
      </div>
    </div>

    <div className="grid grid-cols-2 gap-4">
      <div className="bg-purple-50 border-3 border-purple-500 rounded-xl p-5">
        <div className="flex items-center gap-2 mb-3">
          <Rocket className="w-8 h-8 text-purple-600" />
          <h3 className="text-xl font-bold text-purple-700">2️⃣ SageMaker Integration</h3>
        </div>
        <div className="bg-white p-4 rounded-lg text-sm space-y-2">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-purple-600 rounded-full"></div>
            <span>Train in SageMaker (AWS distributed)</span>
          </div>
          <div className="flex items-center gap-2">
            <ArrowRight className="w-4 h-4 text-gray-400" />
            <span className="text-gray-500">Export trained model</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-purple-600 rounded-full"></div>
            <span>Deploy via AI Catalyst (auto-scaling)</span>
          </div>
          <div className="flex items-center gap-2">
            <ArrowRight className="w-4 h-4 text-gray-400" />
            <span className="text-gray-500">Production endpoint &lt;200ms</span>
          </div>
        </div>
        <div className="mt-3 bg-purple-100 p-3 rounded-lg text-center">
          <p className="text-sm font-semibold text-purple-700">Impact: $20M-$60M</p>
          <p className="text-xs text-gray-600">AWS co-selling</p>
        </div>
      </div>

      <div className="bg-blue-50 border-3 border-blue-500 rounded-xl p-5">
        <div className="flex items-center gap-2 mb-3">
          <Cloud className="w-8 h-8 text-blue-600" />
          <h3 className="text-xl font-bold text-blue-700">3️⃣ Azure Marketplace</h3>
        </div>
        <div className="bg-white p-4 rounded-lg text-sm space-y-2">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
            <span>Click "Deploy" in Azure Marketplace</span>
          </div>
          <div className="flex items-center gap-2">
            <ArrowRight className="w-4 h-4 text-gray-400" />
            <span className="text-gray-500">30 minutes setup</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
            <span>Anaconda running in customer VNet</span>
          </div>
          <div className="flex items-center gap-2">
            <ArrowRight className="w-4 h-4 text-gray-400" />
            <span className="text-gray-500">Billed through Azure contract</span>
          </div>
        </div>
        <div className="mt-3 bg-blue-100 p-3 rounded-lg text-center">
          <p className="text-sm font-semibold text-blue-700">Impact: $30M-$90M</p>
          <p className="text-xs text-gray-600">Azure enterprise wins</p>
        </div>
      </div>
    </div>

    <div className="bg-green-100 border-2 border-green-600 p-5 rounded-lg">
      <h3 className="text-xl font-bold text-green-700 mb-3 text-center">Combined Q1 2026 Impact</h3>
      <div className="grid grid-cols-3 gap-3 text-center">
        <div className="bg-white p-3 rounded-lg">
          <p className="text-sm text-gray-600">TAM Expansion</p>
          <p className="text-2xl font-bold text-green-600">2-3x</p>
        </div>
        <div className="bg-white p-3 rounded-lg">
          <p className="text-sm text-gray-600">New Revenue</p>
          <p className="text-2xl font-bold text-green-600">$100M-$300M</p>
        </div>
        <div className="bg-white p-3 rounded-lg">
          <p className="text-sm text-gray-600">Deployment Time</p>
          <p className="text-2xl font-bold text-green-600">6mo → 1 day</p>
        </div>
      </div>
    </div>
  </div>
</div>
);
// SLIDE 20 - 2026 Q2 Enhancements
slides.push(
<div className="h-full flex flex-col p-12 bg-white">
<Header num={20} />
<h1 className="text-5xl font-bold mb-6 text-gray-900">Q2 2026: Production Hardening + In-Database AI</h1>
  <div className="space-y-5">
    <div className="bg-gradient-to-r from-cyan-500 to-cyan-600 text-white p-6 rounded-xl shadow-lg">
      <div className="flex items-center gap-3 mb-3">
        <Database className="w-10 h-10" />
        <h3 className="text-3xl font-bold">4️⃣ Snowflake Snowpark Integration</h3>
      </div>
      <div className="bg-white/10 backdrop-blur p-4 rounded-lg">
        <div className="bg-gray-900 text-green-400 font-mono text-sm p-4 rounded-lg mb-3">
          <div className="text-gray-400"># Fraud detection INSIDE Snowflake</div>
          <div>@snowpark_udf</div>
          <div>def score_fraud(merchant: str, amount: float):</div>
          <div className="ml-4">return fraud_model.predict_proba(...)</div>
          <div className="text-green-400 mt-2"># Zero data movement, &lt;5ms latency</div>
        </div>
        <div className="grid grid-cols-3 gap-3 text-center">
          {[
            { label: 'Latency', value: '<5ms', sub: '10-40x faster' },
            { label: 'Data Movement', value: 'Zero', sub: 'PCI-DSS compliant' },
            { label: 'Impact', value: '$50M-$150M', sub: 'Regulated markets' }
          ].map((stat, i) => (
            <div key={i} className="bg-white text-gray-800 p-3 rounded-lg">
              <p className="text-xs text-gray-600">{stat.label}</p>
              <p className="text-xl font-bold text-cyan-600">{stat.value}</p>
              <p className="text-xs text-gray-500">{stat.sub}</p>
            </div>
          ))}
        </div>
      </div>
    </div>

    <div className="grid grid-cols-2 gap-4">
      <div className="bg-gray-50 border-3 border-gray-600 rounded-xl p-5">
        <div className="flex items-center gap-2 mb-3">
          <Box className="w-8 h-8 text-gray-700" />
          <h3 className="text-xl font-bold text-gray-800">5️⃣ Docker Hardened Images</h3>
        </div>
        <div className="bg-gray-900 text-green-400 font-mono text-xs p-3 rounded-lg mb-3">
          <div>FROM anaconda/hardened-python:</div>
          <div className="ml-4">3.11-cuda12.1</div>
          <div className="text-green-400 mt-2"># ✓ CVE-scanned base + CUDA</div>
          <div className="text-green-400"># ✓ $10M indemnification</div>
          <div className="text-green-400"># ✓ Production-secure default</div>
        </div>
        <div className="bg-gray-100 p-3 rounded-lg text-center">
          <p className="text-sm font-semibold">Impact: $10M-$30M</p>
          <p className="text-xs text-gray-600">DevOps/security buyers</p>
        </div>
      </div>

      <div className="bg-green-50 border-3 border-green-600 rounded-xl p-5">
        <div className="flex items-center gap-2 mb-3">
          <Zap className="w-8 h-8 text-green-700" />
          <h3 className="text-xl font-bold text-green-800">6️⃣ NVIDIA Partnership</h3>
        </div>
        <div className="bg-white p-4 rounded-lg mb-3 space-y-2 text-sm">
          <div className="flex items-start gap-2">
            <CheckCircle className="w-5 h-5 text-green-600 mt-0.5" />
            <span><strong>Optimized CUDA packages:</strong> 5x faster download</span>
          </div>
          <div className="flex items-start gap-2">
            <CheckCircle className="w-5 h-5 text-green-600 mt-0.5" />
            <span><strong>Guaranteed compatibility:</strong> Pre-tested combinations</span>
          </div>
          <div className="flex items-start gap-2">
            <CheckCircle className="w-5 h-5 text-green-600 mt-0.5" />
            <span><strong>Zero configuration:</strong> conda install pytorch-cuda12.1 → GPU ready</span>
          </div>
        </div>
        <div className="bg-green-100 p-3 rounded-lg text-center">
          <p className="text-sm font-semibold">Impact: $15M-$45M</p>
          <p className="text-xs text-gray-600">High-performance tier</p>
        </div>
      </div>
    </div>

    <div className="bg-gradient-to-r from-green-100 to-green-200 border-2 border-green-600 p-5 rounded-xl">
      <h3 className="text-2xl font-bold text-green-800 mb-3 text-center">Q2 2026 Combined Revenue Impact</h3>
      <div className="grid grid-cols-4 gap-3">
        {[
          { label: 'Snowpark Integration', value: '$50M-$150M' },
          { label: 'Docker Images', value: '$10M-$30M' },
          { label: 'NVIDIA Partnership', value: '$15M-$45M' },
          { label: 'Total Q2 TAM', value: '$75M-$225M' }
        ].map((item, i) => (
          <div key={i} className="bg-white p-3 rounded-lg text-center border-2 border-green-400">
            <p className="text-xs text-gray-600 mb-1">{item.label}</p>
            <p className="text-lg font-bold text-green-700">{item.value}</p>
          </div>
        ))}
      </div>
    </div>
  </div>
</div>
);
// SLIDE 21 - 2026 Q4 Future
slides.push(
<div className="h-full flex flex-col p-12 bg-white">
<Header num={21} />
<h1 className="text-5xl font-bold mb-6 text-gray-900">Q4 2026: Agents + Guardrails</h1>
<h2 className="text-3xl text-gray-600 mb-8">Next-Generation Fraud Detection</h2>
  <div className="grid grid-cols-2 gap-6">
    <div className="bg-blue-50 border-4 border-blue-500 rounded-xl p-6">
      <h3 className="text-2xl font-bold text-blue-700 mb-4 text-center">Phase 1 (2025): Fraud Detection Model</h3>
      <div className="bg-white p-5 rounded-lg border-2 border-blue-300">
        <p className="font-semibold mb-3 text-gray-700">Current Approach:</p>
        <div className="bg-gray-900 text-green-400 font-mono text-sm p-4 rounded-lg">
          <div className="text-gray-400"># Input: Transaction data</div>
          <div>score, explanation = </div>
          <div className="ml-4">fraud_model.predict(</div>
          <div className="ml-8">transaction</div>
          <div className="ml-4">)</div>
          <div className="mt-3 text-gray-400"># Output: Score + explanation</div>
        </div>
        <div className="mt-4 p-3 bg-blue-50 rounded-lg">
          <p className="text-sm"><strong>Capability:</strong> Reactive fraud scoring</p>
          <p className="text-xs text-gray-600 mt-1">Analyzes single transaction, returns verdict</p>
        </div>
      </div>
    </div>

    <div className="bg-green-50 border-4 border-green-600 rounded-xl p-6">
      <h3 className="text-2xl font-bold text-green-700 mb-4 text-center">Phase 2 (Q4 2026): Fraud Investigation Agent</h3>
      <div className="bg-white p-5 rounded-lg border-2 border-green-400">
        <p className="font-semibold mb-3 text-gray-700">Agentic Approach:</p>
        <div className="bg-gray-900 text-green-400 font-mono text-xs p-4 rounded-lg">
          <div className="text-gray-400"># Autonomous investigation</div>
          <div>agent = FraudInvestigator(</div>
          <div className="ml-4">llm=SafeLLM("Qwen2.5-7B",</div>
          <div className="ml-8">guardrails=[</div>
          <div className="ml-12">"no_prompt_injection",</div>
          <div className="ml-12">"no_pii_output",</div>
          <div className="ml-12">"no_discrimination"</div>
          <div className="ml-8">])</div>
          <div className="ml-4">)</div>
          <div className="mt-2">report = agent.investigate(...)</div>
        </div>
        <div className="mt-4 p-3 bg-green-50 rounded-lg">
          <p className="text-sm font-semibold mb-2">Agent Actions:</p>
          <div className="text-xs text-gray-700 space-y-1">
            {['Query transaction history', 'Analyze merchant patterns', 'Check velocity rules', 'Generate investigation report', 'Escalate to human if needed'].map((action, i) => (
              <div key={i} className="flex items-center gap-2">
                <CheckCircle className="w-3 h-3 text-green-600" />
                <span>{action}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  </div>

  <div className="mt-6 bg-gradient-to-r from-purple-500 to-purple-600 text-white p-6 rounded-xl">
    <div className="grid grid-cols-3 gap-6">
      <div className="text-center">
        <TrendingUp className="w-12 h-12 mx-auto mb-2" />
        <p className="text-sm mb-1">Manual Review Reduction</p>
        <p className="text-3xl font-bold">40-60%</p>
        <p className="text-xs mt-1 opacity-90">$500K-$2M operational savings</p>
      </div>
      <div className="text-center">
        <Shield className="w-12 h-12 mx-auto mb-2" />
        <p className="text-sm mb-1">Guardrails Protection</p>
        <p className="text-3xl font-bold">$5M-$20M</p>
        <p className="text-xs mt-1 opacity-90">Avoided bias violations</p>
      </div>
      <div className="text-center">
        <Award className="w-12 h-12 mx-auto mb-2" />
        <p className="text-sm mb-1">Market Leadership</p>
        <p className="text-3xl font-bold">$30M-$100M</p>
        <p className="text-xs mt-1 opacity-90">Future-proof positioning</p>
      </div>
    </div>
  </div>
</div>
);
    Architecture:
        Stage 1: XGBoost - Fast screening of all transactions
        Stage 2: Qwen 2.5 7B - Deep analysis of high-risk cases
        
    Advantages:
        - Speed: XGBoost handles bulk screening (ms per transaction)
        - Accuracy: LLM catches nuanced fraud patterns
        - Efficiency: LLM only runs on high-risk subset
        
    Personas:
        - Sarah: Trains and tunes the model
        - Marcus: Deploys to production via AI Catalyst
        - Michael: Validates business outcomes
        
    Anaconda Value:
        - Automatic dependency tracking (sklearn + torch)
        - Reproducible hybrid architecture
        - One-click deployment to production
    """
    
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
            print(f"  ✓ Stage 2: Analyzed {llm_analyzed} cases in {elapsed:.1f}s")
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