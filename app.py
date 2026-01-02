# pylint: disable=redefined-outer-name
"""
Fraud Detection Dashboard - Streamlit Application

Launch with: streamlit run app.py

Features:
- Real-time fraud detection testing
- Interactive transaction analysis
- Performance monitoring
- System status checks
- Production API integration
"""

import streamlit as st
import numpy as np
#import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
#import time
import random

from src.config import (
    CONNECT_ENDPOINT, NAVIGATOR_ENDPOINT,
    LEGITIMATE_MERCHANTS, SUSPICIOUS_MERCHANTS
)
from src.api_client import FraudDetectionAPI

# ================================================================================
# PAGE CONFIGURATION
# ================================================================================

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================================
# CUSTOM CSS
# ================================================================================

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================================
# INITIALIZE API CLIENT
# ================================================================================

@st.cache_resource
def get_api_client():
    return FraudDetectionAPI(CONNECT_ENDPOINT, NAVIGATOR_ENDPOINT)

api_client = get_api_client()

# ================================================================================
# HELPER FUNCTIONS
# ================================================================================

def render_business_result(result, merchant: str, amount: float):
    """Render fraud detection result"""
    prob = float(result.get("probability", 0.0))
    pred = int(result.get("prediction", 0))
    source = result.get("source", "Unknown")
    latency = float(result.get("latency_ms", 0.0))

    risk_pct = round(prob * 100)

    if prob < 0.30:
        band = "Low"
        band_msg = "Looks consistent with normal customer behavior."
    elif prob < 0.60:
        band = "Medium"
        band_msg = "Some risk signals present."
    else:
        band = "High"
        band_msg = "Strong fraud indicators."

    st.subheader("Decision Summary")

    if pred == 1:
        st.error("Recommended action: **Flag for review**")
    else:
        st.success("Recommended action: **Approve**")

    col1, col2, col3 = st.columns(3)
    col1.metric("Fraud Risk Score", f"{risk_pct}%")
    col2.metric("Risk Level", band)
    col3.metric("Response Time", f"{latency:.0f} ms")

    st.progress(min(max(prob, 0.0), 1.0))

    st.markdown("### Transaction")
    st.write(f"**Merchant:** {merchant}")
    st.write(f"**Amount:** ${amount:,.2f}")

    st.markdown("### Explanation")
    st.write(band_msg)

    with st.expander("Technical Details"):
        st.write(f"**Source:** {source}")
        st.write(f"**Probability:** {prob:.4f}")

# ================================================================================
# SIDEBAR
# ================================================================================

st.sidebar.title("Fraud Detection System")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Test Transaction", "Analytics", "System Status"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Powered By")
st.sidebar.markdown("**Anaconda Platform**")
st.sidebar.markdown("- Core: Package Management")
st.sidebar.markdown("- Desktop: Development")
st.sidebar.markdown("- AI Catalyst: Deployment")

# ================================================================================
# PAGE 1: DASHBOARD
# ================================================================================

if page == "Dashboard":
    st.title("Real-Time Fraud Detection Dashboard")
    st.markdown("### Powered by Anaconda AI Catalyst")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Transactions Today", "12,547", "↑ 8.2%")
    with col2:
        st.metric("Fraud Detected", "23", "↓ 12.5%", delta_color="inverse")
    with col3:
        st.metric("Fraud Rate", "0.18%", "↓ 0.05%", delta_color="inverse")
    with col4:
        st.metric("Avg Latency", "42ms", "↓ 5ms", delta_color="inverse")

    st.markdown("---")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Real-Time Activity")

        hours = list(range(24))
        transactions = np.random.randint(400, 600, 24).tolist()
        fraud = np.random.randint(0, 5, 24).tolist()

        fig = go.Figure()
        fig.add_trace(go.Bar(x=hours, y=transactions, name='Total', marker_color='#667eea'))
        fig.add_trace(go.Bar(x=hours, y=fraud, name='Fraud', marker_color='#ff4444'))
        fig.update_layout(title="Last 24 Hours", xaxis_title="Hour", 
                         yaxis_title="Count", barmode='group', height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Model Performance")

        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        scores = [0.9985, 0.9421, 0.8734, 0.9063]
        colors = ['#00C851' if s > 0.85 else '#ffbb33' for s in scores]

        fig = go.Figure(go.Bar(
            x=scores, y=metrics, orientation='h',
            marker_color=colors,
            text=[f'{s:.2%}' for s in scores],
            textposition='auto'
        ))
        fig.update_layout(title="Current Metrics", height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ================================================================================
# PAGE 2: TEST TRANSACTION
# ================================================================================

elif page == "Test Transaction":
    st.title("Test Transaction Analysis")

    if "transaction_type" not in st.session_state:
        st.session_state.transaction_type = "Legitimate Purchase"
    if "merchant" not in st.session_state:
        st.session_state.merchant = LEGITIMATE_MERCHANTS[0]
    if "amount" not in st.session_state:
        st.session_state.amount = 67.89

    def set_random_legit():
        st.session_state.merchant = random.choice(LEGITIMATE_MERCHANTS)
        st.session_state.amount = round(random.uniform(10, 300), 2)

    def set_random_susp():
        st.session_state.merchant = random.choice(SUSPICIOUS_MERCHANTS)
        st.session_state.amount = round(random.uniform(800, 5000), 2)

    col_left, col_right = st.columns([3, 1])

    with col_left:
        st.subheader("Transaction Details")

        transaction_type = st.radio(
            "Select Transaction Type",
            ["Legitimate Purchase", "Suspicious Activity", "Custom"]
        )

        if transaction_type == "Legitimate Purchase":
            merchant = st.selectbox("Merchant", LEGITIMATE_MERCHANTS)
            amount = st.slider("Amount ($)", 10.0, 500.0, float(st.session_state.amount))
        elif transaction_type == "Suspicious Activity":
            merchant = st.selectbox("Merchant", SUSPICIOUS_MERCHANTS)
            amount = st.slider("Amount ($)", 500.0, 5000.0, float(st.session_state.amount))
        else:
            merchant = st.text_input("Merchant", st.session_state.merchant)
            amount = st.number_input("Amount ($)", min_value=1.0, value=float(st.session_state.amount))

        st.session_state.merchant = merchant
        st.session_state.amount = amount

        if st.button("Analyze Transaction", type="primary"):
            with st.spinner("Analyzing..."):
                result = api_client.predict(merchant, amount)
     
            if result.get("success"):
                render_business_result(result, merchant, amount)
            else:
                st.error(f"Error: {result.get('error', 'Unknown')}")

    with col_right:
        st.subheader("Quick Actions")
        st.button("Random Legitimate", on_click=set_random_legit)
        st.button("Random Suspicious", on_click=set_random_susp)

# ================================================================================
# PAGE 3: ANALYTICS
# ================================================================================

elif page == "Analytics":
    st.title("Advanced Analytics")
    
    time_range = st.selectbox("Time Range", ["Last 24 Hours", "Last 7 Days", "Last 30 Days"])
    
    st.subheader(" Fraud by Category")
    
    categories = ['Online Shopping', 'Gas Stations', 'Restaurants', 'Crypto/ATM', 'Wire Transfers']
    fraud_counts = [5, 2, 1, 8, 4]
    
    fig = px.pie(values=fraud_counts, names=categories, title="Fraud Cases")
    st.plotly_chart(fig, use_container_width=True)

# ================================================================================
# PAGE 4: SYSTEM STATUS
# ================================================================================

elif page == "System Status":
    st.title("System Status & Monitoring")
    
    st.subheader("API Connection Test")
    
    if st.button("Test API Connections"):
        with st.spinner("Testing..."):
            results = api_client.test_connection()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if results['connect']:
                st.success("Anaconda Connect")
            else:
                st.warning("Connect Offline")
        
        with col2:
            if results['navigator']:
                st.success("AI Navigator")
            else:
                st.warning("Navigator Offline")
        
        with col3:
            st.success("Mock Fallback")

# ================================================================================
# FOOTER
# ================================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Powered by Anaconda Platform</strong></p>
    <p>Core - Desktop - AI Catalyst</p>
    <p>2025 Fraud Detection System</p>
</div>
""", unsafe_allow_html=True)