"""
Data Loading and Feature Engineering Utilities

This module provides functions for:
- Loading fraud detection dataset
- Generating merchant descriptions
- Feature engineering
- Train/test splitting

Persona: Sarah Chen (Data Scientist)
Anaconda Value: Integrated data engineering workflow
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from src.config import (
    DATA_PATH, LEGITIMATE_MERCHANTS, SUSPICIOUS_MERCHANTS,
    TEST_SIZE, RANDOM_STATE, TRAIN_SAMPLE_SIZE, TEST_SAMPLE_SIZE
)

# ================================================================================
# DATA LOADING
# ================================================================================

def load_fraud_data(filepath: Path = DATA_PATH, verbose: bool = True) -> pd.DataFrame:
    """
    Load fraud detection dataset.

    Args:
        filepath: Path to creditcard.csv (default: DATA_PATH from config)
        verbose: Print loading statistics

    Returns:
        pandas DataFrame with fraud transaction data
    """
    if not filepath.exists():
        raise FileNotFoundError(
            "Dataset not found.\n"
            "Download from Kaggle and place the file at:\n"
            f"{filepath}\n\n"
            "See data/README.md for instructions."
        )

    data = pd.read_csv(filepath)

    if verbose:
        print(f" Dataset loaded: {len(data):,} transactions")
        print(f"  • Features: {data.shape[1]} columns")
        print(f"  • Fraud rate: {(data['Class'] == 1).mean() * 100:.4f}%")
        legit = (data['Class'] == 0).sum()
        fraud = (data['Class'] == 1).sum()
        print(f"  • Imbalance ratio: {legit / fraud:.0f}:1")

    return data


# ================================================================================
# MERCHANT DESCRIPTION GENERATION
# ================================================================================

def generate_merchant_description(is_fraud):
    """
    Generate realistic merchant name based on fraud status
    
    Args:
        is_fraud: Boolean indicating if transaction is fraudulent
        
    Returns:
        str: Merchant description
        
    Business Context:
        Legitimate merchants: Major retailers, known brands
        Suspicious merchants: Crypto, wire transfers, unknown sources
    """
    merchants = SUSPICIOUS_MERCHANTS if is_fraud else LEGITIMATE_MERCHANTS
    template = np.random.choice(merchants)
    
    # Add random numbers to templates with placeholders
    if '{:04d}' in template:
        return template.format(np.random.randint(1000, 9999))
    return template


def add_merchant_descriptions(data, verbose=True):
    """
    Add merchant_description column to dataset
    
    Args:
        data: DataFrame with 'Class' column
        verbose: Print progress
        
    Returns:
        DataFrame with new 'merchant_description' column
        
    Anaconda Value:
        - Multi-modal feature engineering (numeric + text)
        - Prepares data for hybrid ML+LLM model
    """
    if verbose:
        print("\n Engineering text features for LLM analysis...")
    
    np.random.seed(RANDOM_STATE)
    data['merchant_description'] = [
        generate_merchant_description(c == 1) 
        for c in data['Class']
    ]
    
    if verbose:
        print(" Text features added")
        print(f"  • Sample legitimate: {data[data['Class']==0]['merchant_description'].iloc[0]}")
        print(f"  • Sample fraud: {data[data['Class']==1]['merchant_description'].iloc[0]}")
    
    return data


# ================================================================================
# DATA SPLITTING
# ================================================================================

def prepare_train_test_split(data, demo_mode=True, verbose=True):
    """
    Split data into training and test sets with optional sampling
    
    Args:
        data: Full DataFrame with features, Class, merchant_description
        demo_mode: If True, apply sampling for faster demos
        verbose: Print split statistics
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, desc_train, desc_test, 
                  amt_train, amt_test)
    
    Demo Optimization:
        - Demo mode: 50K train, 10K test (5-10 min runtime)
        - Full mode: All data (60+ min runtime)
        
    Anaconda Value:
        - Reproducible splits via random_state
        - Stratified sampling maintains fraud rate
    """
    if verbose:
        print("\n Splitting data for training and testing")
        print("=" * 70)
    
    # Separate features and targets
    X = data.drop(['Class', 'merchant_description'], axis=1)
    y = data['Class']
    descriptions = data['merchant_description']
    amounts = data['Amount']
    
    # Stratified split to maintain fraud rate
    X_train, X_test, y_train, y_test, desc_train, desc_test, amt_train, amt_test = train_test_split(
        X, y, descriptions, amounts,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    # Apply demo mode sampling if enabled
    if demo_mode and TRAIN_SAMPLE_SIZE:
        if verbose:
            print(f" Demo mode enabled - sampling for fast execution")
        
        # Sample training data
        train_idx = np.random.choice(len(X_train), 
                                     min(TRAIN_SAMPLE_SIZE, len(X_train)), 
                                     replace=False)
        X_train = X_train.iloc[train_idx]
        y_train = y_train.iloc[train_idx]
        desc_train = desc_train.iloc[train_idx]
        amt_train = amt_train.iloc[train_idx]
        
        # Sample test data
        test_idx = np.random.choice(len(X_test), 
                                    min(TEST_SAMPLE_SIZE, len(X_test)), 
                                    replace=False)
        X_test = X_test.iloc[test_idx]
        y_test = y_test.iloc[test_idx]
        desc_test = desc_test.iloc[test_idx]
        amt_test = amt_test.iloc[test_idx]
    
    if verbose:
        print(f"\n Data split complete:")
        print(f"  • Training: {len(X_train):,} transactions ({(y_train==1).sum()} fraud)")
        print(f"  • Testing: {len(X_test):,} transactions ({(y_test==1).sum()} fraud)")
        print(f"  • Train fraud rate: {(y_train==1).sum()/len(y_train)*100:.4f}%")
        print(f"  • Test fraud rate: {(y_test==1).sum()/len(y_test)*100:.4f}%")
        print("=" * 70)
    
    return X_train, X_test, y_train, y_test, desc_train, desc_test, amt_train, amt_test


# ================================================================================
# FEATURE GENERATION FOR DEMOS
# ================================================================================

def generate_realistic_features(merchant, amount, is_suspicious=False, reference_data=None):
    """
    Generate realistic feature vector for demo transactions
    
    Args:
        merchant: Merchant description
        amount: Transaction amount
        is_suspicious: Whether to use fraud-like features
        reference_data: Optional DataFrame to sample from
        
    Returns:
        numpy array of 30 features (28 PCA + Time + Amount)
        
    Use Case:
        For interactive demos and testing without real data
    """
    if reference_data is not None:
        # Use actual transaction features
        if is_suspicious:
            fraud_txns = reference_data[reference_data['Class'] == 1]
            if len(fraud_txns) > 0:
                sample = fraud_txns.sample(1, random_state=RANDOM_STATE)
                features = sample.drop(['Class', 'merchant_description'], axis=1).values[0]
                features[-1] = amount  # Update amount
                return features
        else:
            legit_txns = reference_data[reference_data['Class'] == 0]
            # Find similar amount range
            similar = legit_txns[
                (legit_txns['Amount'] >= amount * 0.8) & 
                (legit_txns['Amount'] <= amount * 1.2)
            ]
            if len(similar) > 0:
                sample = similar.sample(1, random_state=RANDOM_STATE)
                features = sample.drop(['Class', 'merchant_description'], axis=1).values[0]
                features[-1] = amount
                return features
    
    # Fallback: generate synthetic features
    np.random.seed(int(amount * 1000) % 2**32)
    if is_suspicious:
        features = np.random.randn(28) * 3  # High variance
    else:
        features = np.random.randn(28) * 0.5  # Low variance
    
    # Add Time and Amount
    features = np.append(features, [np.random.randint(0, 172800), amount])
    return features


# ================================================================================
# DATA VALIDATION
# ================================================================================

def validate_data(data, verbose=True):
    """
    Validate fraud dataset has required structure
    
    Args:
        data: DataFrame to validate
        verbose: Print validation results
        
    Returns:
        bool: True if valid, raises ValueError if not
        
    Required Columns:
        - Class: 0 (legitimate) or 1 (fraud)
        - Amount: Transaction amount
        - V1-V28: PCA components
        - Time: Seconds since first transaction
    """
    required_cols = ['Class', 'Amount', 'Time'] + [f'V{i}' for i in range(1, 29)]
    missing = [col for col in required_cols if col not in data.columns]
    
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    if not set(data['Class'].unique()).issubset({0, 1}):
        raise ValueError("Class column must contain only 0 (legit) and 1 (fraud)")
    
    if verbose:
        print(" Data validation passed")
        print(f"  • All {len(required_cols)} required columns present")
        print(f"  • Class labels valid (0, 1)")
    
    return True


# ================================================================================
# SUMMARY STATISTICS
# ================================================================================

def print_data_summary(data):
    """
    Print comprehensive data summary for exploratory analysis
    
    Persona: Sarah (Data Scientist) - Initial exploration
    """
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    
    print(f"\nShape: {data.shape}")
    print(f"Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\nClass Distribution:")
    print(data['Class'].value_counts())
    print(f"Fraud rate: {data['Class'].mean()*100:.4f}%")
    
    print("\nAmount Statistics:")
    print(data['Amount'].describe())
    
    print("\nMissing Values:")
    missing = data.isnull().sum()
    if missing.sum() == 0:
        print("   No missing values")
    else:
        print(missing[missing > 0])
    
    print("=" * 70)