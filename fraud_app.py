"""
Fraud Detection Dashboard - Streamlit Application

Launch with: streamlit run app.py

Features:
- Real-time fraud detection testing
- Interactive transaction analysis
- Performance monitoring
- System status checks
- Production API integration

Anaconda Value:
- Built with Anaconda-managed packages
- Connects to AI Catalyst deployed models
- Production-ready interface
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import random

# Import configuration
from src.config import (
    CONNECT_ENDPOINT, NAVIGATOR_ENDPOINT,
    LEGITIMATE_MERCHANTS, SUSPICIOUS_MERCHANTS
)

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
        margin: 10px 0;
    }
    .fraud-alert {
        background-color: #ff4444;
        padding: 15px;
        border-radius: 5px;
        color: white;
        font-weight: bold;
    }
    .safe-transaction {
        background-color: #00C851;
        padding: 15px;
        border-radius: 5px;
        color: white;
        font-weight: bold;
    }
    .warning-transaction {
        background-color: #ffbb33;
        padding: 15px;
        border-radius: 5px;
        color: white;
        font-weight: bold;
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
    div[data-testid="stMetricValue"] {
        font-size: 28px;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================================
# API CLIENT (EMBEDDED FOR STREAMLIT)
# ================================================================================

import requests
import json

class FraudDetectionAPI:
    """
    Multi-endpoint fraud detection API client with automatic fallback
    """
    
    def __init__(self, connect_endpoint: str, navigator_endpoint: str):
        self.connect_endpoint = connect_endpoint
        self.navigator_endpoint = navigator_endpoint
        self.session = requests.Session()
        self.last_source = "Not Used Yet"

    def predict(self, merchant, amount, features=None):
        if features is None:
            features = self._generate_features(merchant, amount)

        start_time = time.time()

        # 1) Try Anaconda Connect deployed model first
        r = self._try_connect_inference(merchant, amount, features)
        if r is not None:
            r["latency_ms"] = (time.time() - start_time) * 1000
            r["timestamp"] = datetime.now()
            r["source"] = "Anaconda Connect (Deployed Model)"
            self.last_source = r["source"]
            return r

        # 2) Fallback to Local AI Navigator
        r = self._try_navigator_llm(merchant, amount)
        if r is not None:
            r["latency_ms"] = (time.time() - start_time) * 1000
            r["timestamp"] = datetime.now()
            r["source"] = "AI Navigator (Local)"
            self.last_source = r["source"]
            return r

        # 3) Final fallback: mock heuristic
        latency = (time.time() - start_time) * 1000
        r = self._mock_predict(merchant, amount, features, latency)
        self.last_source = "Mock Model (Fallback)"
        return r

    def _try_connect_inference(self, merchant, amount, features):
        payload = {
            "data": [features.tolist()],
            "merchant_description": [merchant],
            "amount": [float(amount)]
        }
        try:
            resp = self.session.post(self.connect_endpoint, json=payload, timeout=10)
            if resp.status_code != 200:
                return None
            result = resp.json()
            prob = result.get("probability", [0.5])[0]
            pred = result.get("prediction", [1 if prob >= 0.5 else 0])[0]
            return {"success": True, "prediction": int(pred), "probability": float(prob)}
        except Exception:
            return None

    def _try_navigator_llm(self, merchant, amount):
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
            resp = self.session.post(self.navigator_endpoint, json=payload, timeout=10)
            if resp.status_code != 200:
                return None
            out = resp.json()
            content = out["choices"][0]["message"]["content"]
            data = json.loads(content)
            prob = float(data["probability"])
            prob = max(0.0, min(1.0, prob))
            pred = 1 if prob >= 0.5 else 0
            return {"success": True, "prediction": pred, "probability": prob}
        except Exception:
            return None

    def _mock_predict(self, merchant, amount, features, latency):
        merchant_upper = merchant.upper()
        suspicious_keywords = ['BITCOIN', 'CRYPTO', 'CASINO', 'WIRE', 'FOREIGN', 'UNKNOWN', 'UNVERIFIED']
        is_suspicious = any(k in merchant_upper for k in suspicious_keywords)

        base_prob = 0.75 if is_suspicious else 0.15
        if amount > 2000:
            amount_factor = 0.15
        elif amount > 1000:
            amount_factor = 0.10
        elif amount > 500:
            amount_factor = 0.05
        else:
            amount_factor = 0.0

        probability = min(base_prob + amount_factor + np.random.uniform(-0.05, 0.05), 0.99)
        probability = max(probability, 0.01)
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
        np.random.seed(int(time.time() * 1000) % 2**32)
        if any(s in merchant.upper() for s in ["BITCOIN", "CRYPTO", "CASINO", "WIRE", "FOREIGN"]):
            features = np.random.randn(28) * 3
        else:
            features = np.random.randn(28) * 0.5
        features = np.append(features, [np.random.randint(0, 172800), amount])
        return features

    def test_connection(self):
        """Test connectivity to all endpoints"""
        results = {
            'connect': False,
            'navigator': False,
            'mock': True
        }
        
        # Test Connect
        try:
            test_payload = {
                "data": [[0] * 30],
                "merchant_description": ["TEST"],
                "amount": [100.0]
            }
            resp = self.session.post(self.connect_endpoint, json=test_payload, timeout=5)
            results['connect'] = resp.status_code in [200, 400]
        except:
            pass
        
        # Test Navigator
        try:
            test_payload = {
                "messages": [{"role": "user", "content": "test"}],
                "temperature": 0.0
            }
            resp = self.session.post(self.navigator_endpoint, json=test_payload, timeout=5)
            results['navigator'] = resp.status_code in [200, 400]
        except:
            pass
        
        return results


# ================================================================================
# HELPER FUNCTIONS
# ================================================================================

def render_business_result(result, merchant: str, amount: float):
    """Render fraud detection result in business-friendly format"""
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
        band_msg = "Some risk signals present. Recommend a light review."
    else:
        band = "High"
        band_msg = "Strong fraud indicators. Recommend holding or step-up verification."

    st.subheader("Decision Summary")

    if pred == 1:
        st.error("Recommended action: **Flag for review**")
    else:
        st.success("Recommended action: **Approve**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud Risk Score", f"{risk_pct}%")
    c2.metric("Risk Level", band)
    c3.metric("Response Time", f"{latency:.0f} ms")

    st.caption("Fraud risk score (0% = low, 100% = high)")
    st.progress(min(max(prob, 0.0), 1.0))

    st.markdown("### Transaction")
    st.write(f"**Merchant:** {merchant}")
    st.write(f"**Amount:** ${amount:,.2f}")

    st.markdown("### Explanation")
    st.write(band_msg)

    st.markdown("### Suggested Next Steps")
    if prob < 0.30:
        st.write("- Proceed normally")
        st.write("- Monitor if multiple similar transactions occur")
    elif prob < 0.60:
        st.write("- Confirm customer identity (OTP / verification)")
        st.write("- Review recent purchase history")
        st.write("- Contact customer if pattern looks unusual")
    else:
        st.write("- Hold or block pending verification")
        st.write("- Require step-up authentication")
        st.write("- Escalate to fraud operations for review")

    with st.expander("Technical Details (for analysts)"):
        st.write(f"**Model source:** {source}")
        st.write(f"**Raw probability:** {prob:.4f}")
        st.write(f"**Raw prediction:** {pred} (1=fraud, 0=legit)")


# ================================================================================
# INITIALIZE API CLIENT
# ================================================================================

@st.cache_resource
def get_api_client():
    return FraudDetectionAPI(CONNECT_ENDPOINT, NAVIGATOR_ENDPOINT)

api_client = get_api_client()

# ================================================================================
# SIDEBAR
# ================================================================================

st.sidebar.title("Fraud Detection System")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", ["Dashboard", "Test Transaction", "Analytics", "System Status"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Powered By")
st.sidebar.markdown("**Anaconda Platform**")
st.sidebar.markdown("- Core: Package Management")
st.sidebar.markdown("- Desktop: Development")
st.sidebar.markdown("- AI Catalyst: Deployment")

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Info")
st.sidebar.markdown("**Status:** Running")
st.sidebar.markdown("**Endpoint:** AI Navigator (Local)")
st.sidebar.markdown("**Model:** Qwen 2.5 7B")
st.sidebar.markdown("**Port:** 8080")
st.sidebar.markdown("**Status:** Live")
st.sidebar.markdown("**Endpoint:** Anaconda Connect")
st.sidebar.markdown("**Model:** Hybrid XGBoost + Qwen 2.5")

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
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(" Real-Time Activity")
        hours = list(range(24))
        transactions = np.random.randint(400, 600, 24)
        fraud = np.random.randint(0, 5, 24)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=hours, y=transactions, name='Total Transactions', marker_color='#667eea'))
        fig.add_trace(go.Bar(x=hours, y=fraud, name='Fraud Detected', marker_color='#ff4444'))
        fig.update_layout(title="Transaction Volume (Last 24 Hours)", xaxis_title="Hour", 
                        yaxis_title="Count", barmode='group', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader(" Model Performance")
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        scores = [0.9985, 0.9421, 0.8734, 0.9063]
        
        fig = go.Figure(go.Bar(
            x=scores, y=metrics, orientation='h',
            marker_color=['#00C851' if s > 0.85 else '#ffbb33' for s in scores],
            text=[f'{s:.2%}' for s in scores], textposition='auto'
        ))
        fig.update_layout(title="Current Metrics", xaxis_title="Score", height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader(" Recent Fraud Alerts")
    
    recent_fraud = pd.DataFrame({
        'Time': [(datetime.now() - timedelta(minutes=i*5)).strftime('%H:%M:%S') for i in range(5)],
        'Merchant': np.random.choice(SUSPICIOUS_MERCHANTS, 5),
        'Amount': [f"${x:.2f}" for x in np.random.uniform(500, 5000, 5)],
        'Score': [f"{x:.3f}" for x in np.random.uniform(0.8, 0.99, 5)],
        'Status': ['BLOCKED'] * 5
    })
    st.dataframe(recent_fraud, use_container_width=True)

# ================================================================================
# PAGE 2: TEST TRANSACTION
# ================================================================================

elif page == "Test Transaction":
    st.title("Test Transaction Analysis")
    st.markdown("Try the fraud detection model in real-time")

    # Session state initialization
    if "transaction_type" not in st.session_state:
        st.session_state.transaction_type = "Legitimate Purchase"
    if "merchant" not in st.session_state:
        st.session_state.merchant = LEGITIMATE_MERCHANTS[0]
    if "amount" not in st.session_state:
        st.session_state.amount = 67.89

    # Random button callbacks
    def set_random_legit():
        st.session_state.transaction_type = "Legitimate Purchase"
        st.session_state.merchant = random.choice(LEGITIMATE_MERCHANTS)
        st.session_state.amount = round(random.uniform(10, 300), 2)

    def set_random_susp():
        st.session_state.transaction_type = "Suspicious Activity"
        st.session_state.merchant = random.choice(SUSPICIOUS_MERCHANTS)
        st.session_state.amount = round(random.uniform(800, 5000), 2)

    # Layout
    col_left, col_right = st.columns([3, 1])

    # LEFT COLUMN – Inputs
    with col_left:
        st.subheader("Transaction Details")

        transaction_type = st.radio(
            "Select Transaction Type",
            ["Legitimate Purchase", "Suspicious Activity", "Custom"],
            index=["Legitimate Purchase", "Suspicious Activity", "Custom"].index(
                st.session_state.transaction_type
            ),
            key="transaction_type_radio"
        )
        st.session_state.transaction_type = transaction_type

        # Merchant + Amount
        if transaction_type == "Legitimate Purchase":
            if st.session_state.merchant not in LEGITIMATE_MERCHANTS:
                st.session_state.merchant = LEGITIMATE_MERCHANTS[0]

            merchant = st.selectbox(
                "Merchant",
                LEGITIMATE_MERCHANTS,
                index=LEGITIMATE_MERCHANTS.index(st.session_state.merchant),
                key="merchant_legit"
            )
            amount = st.slider(
                "Amount ($)",
                10.0, 500.0,
                float(st.session_state.amount),
                key="amount_legit"
            )

        elif transaction_type == "Suspicious Activity":
            if st.session_state.merchant not in SUSPICIOUS_MERCHANTS:
                st.session_state.merchant = SUSPICIOUS_MERCHANTS[0]

            merchant = st.selectbox(
                "Merchant",
                SUSPICIOUS_MERCHANTS,
                index=SUSPICIOUS_MERCHANTS.index(st.session_state.merchant),
                key="merchant_susp"
            )
            amount = st.slider(
                "Amount ($)",
                500.0, 5000.0,
                float(st.session_state.amount),
                key="amount_susp"
            )

        else:  # Custom
            merchant = st.text_input("Merchant", st.session_state.merchant, key="merchant_custom")
            amount = st.number_input(
                "Amount ($)", min_value=1.0, value=float(st.session_state.amount), key="amount_custom"
            )

        # Sync state
        st.session_state.merchant = merchant
        st.session_state.amount = amount

        analyze = st.button("Analyze Transaction", type="primary", key="analyze_btn")

        if analyze:
            with st.spinner("Analyzing transaction..."):
                result = api_client.predict(merchant, amount)

            if result.get("success"):
                render_business_result(result, merchant, amount)
            else:
                st.error(result.get("error", "Something went wrong"))

    # RIGHT COLUMN – Random buttons
    with col_right:
        st.subheader("Quick Actions")
        st.button("Random Legitimate", key="rand_legit", on_click=set_random_legit)
        st.button("Random Suspicious", key="rand_susp", on_click=set_random_susp)

# ================================================================================
# PAGE 3: ANALYTICS
# ================================================================================

# ================================================================================
# PAGE 3: ANALYTICS
# ================================================================================

elif page == "Analytics":
    st.title("Advanced Analytics")
    
    time_range = st.selectbox(
        "Select Time Range",
        ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Last 90 Days"]
    )
    
    st.subheader("Fraud Distribution by Category")
    
    # Generate different fraud patterns based on time range
    categories = ['Crypto/ATM', 'Online Shopping', 'Wire Transfers', 'Other', 'Gas Stations', 'Restaurants']
    
    # Different fraud patterns for each time range (simulated realistic data)
    fraud_data_by_range = {
        "Last 24 Hours": [8, 5, 4, 3, 2, 1],  # 23 total frauds (as shown in dashboard)
        "Last 7 Days": [56, 35, 28, 21, 14, 7],  # ~161 total frauds
        "Last 30 Days": [240, 150, 120, 90, 60, 30],  # ~690 total frauds
        "Last 90 Days": [720, 450, 360, 270, 180, 90]  # ~2,070 total frauds
    }
    
    # Get fraud counts for selected time range
    fraud_counts = fraud_data_by_range[time_range]
    total_frauds = sum(fraud_counts)
    
    # Create pie chart with percentages
    fig = px.pie(
        values=fraud_counts,
        names=categories,
        title=f"Fraud Cases by Category ({time_range})",
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    
    # Update layout for better readability
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add summary stats below pie chart
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Fraud Cases", f"{total_frauds:,}", f"{time_range}")
    with col2:
        st.metric("Highest Risk Category", "Crypto/ATM", f"{fraud_counts[0]} cases")
    with col3:
        fraud_rate = total_frauds / (12547 if time_range == "Last 24 Hours" else 
                                     87829 if time_range == "Last 7 Days" else
                                     376320 if time_range == "Last 30 Days" else
                                     1128960) * 100
        st.metric("Fraud Rate", f"{fraud_rate:.2f}%", delta_color="inverse")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚡ Latency Trends")
        
        # Adjust number of days based on selection
        if time_range == "Last 24 Hours":
            x_data = list(range(24))
            x_label = "Hour"
            latency_data = 45 + np.random.randn(24) * 5
        elif time_range == "Last 7 Days":
            x_data = list(range(1, 8))
            x_label = "Day"
            latency_data = 45 + np.random.randn(7) * 5
        elif time_range == "Last 30 Days":
            x_data = list(range(1, 31))
            x_label = "Day"
            latency_data = 45 + np.random.randn(30) * 5
        else:  # Last 90 Days
            x_data = list(range(1, 91))
            x_label = "Day"
            latency_data = 45 + np.random.randn(90) * 5
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_data,
            y=latency_data,
            mode='lines+markers',
            name='Latency',
            line=dict(color='#667eea', width=2),
            hovertemplate=f'<b>{x_label} %{{x}}</b><br>Latency: %{{y:.1f}}ms<extra></extra>'
        ))
        fig.add_hline(
            y=100,
            line_dash="dash",
            line_color="red",
            annotation_text="SLA: 100ms"
        )
        
        fig.update_layout(
            xaxis_title=x_label,
            yaxis_title="Latency (ms)",
            height=300,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add metric summary
        avg_latency = np.mean(latency_data)
        max_latency = np.max(latency_data)
        sla_violations = sum(1 for x in latency_data if x > 100)
        
        st.caption(f"Avg: {avg_latency:.1f}ms | Max: {max_latency:.1f}ms | SLA Violations: {sla_violations}")
    
    with col2:
        st.subheader("📈 Accuracy Trends")
        
        # Generate accuracy data matching time range
        if time_range == "Last 24 Hours":
            accuracy_data = 0.995 + np.random.randn(24) * 0.005
            x_data = list(range(24))
            x_label = "Hour"
        elif time_range == "Last 7 Days":
            accuracy_data = 0.995 + np.random.randn(7) * 0.005
            x_data = list(range(1, 8))
            x_label = "Day"
        elif time_range == "Last 30 Days":
            accuracy_data = 0.995 + np.random.randn(30) * 0.005
            x_data = list(range(1, 31))
            x_label = "Day"
        else:  # Last 90 Days
            accuracy_data = 0.995 + np.random.randn(90) * 0.005
            x_data = list(range(1, 91))
            x_label = "Day"
        
        # Clip to realistic range
        accuracy_data = np.clip(accuracy_data, 0.97, 1.0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_data,
            y=accuracy_data,
            mode='lines+markers',
            name='Accuracy',
            line=dict(color='#00C851', width=2),
            fill='tozeroy',
            hovertemplate=f'<b>{x_label} %{{x}}</b><br>Accuracy: %{{y:.4f}}<extra></extra>'
        ))
        
        fig.update_layout(
            xaxis_title=x_label,
            yaxis_title="Accuracy",
            height=300,
            yaxis=dict(range=[0.97, 1.0]),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add metric summary
        avg_accuracy = np.mean(accuracy_data)
        min_accuracy = np.min(accuracy_data)
        
        st.caption(f"Avg: {avg_accuracy:.4f} | Min: {min_accuracy:.4f} | Target: ≥0.985")
# ================================================================================
# PAGE 4: SYSTEM STATUS
# ================================================================================

elif page == "System Status":
    st.title("System Status & Monitoring")
    
    st.subheader("API Connection Test")
    if st.button("Test AI Navigator API Connection"):
        with st.spinner("Testing connection..."):
            result = api_client.predict("TEST CONNECTION", 100.0)
            
            if result['success']:
                source = result.get('source', 'Unknown')
                if 'Mock' in source:
                    st.warning(f"AI Navigator API is unavailable - Using mock model")
                    st.info(f"Mock response time: {result['latency_ms']:.1f}ms")
                else:
                    st.success(f"API is responding - Latency: {result['latency_ms']:.1f}ms")
                    st.info(f"Connected to: {source}")
    
    st.markdown("---")
    st.subheader("Endpoint Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Production (AI Catalyst):**")
        st.code(CONNECT_ENDPOINT)
        st.markdown("**Hosted on:** AI Navigator (Local Server)")
        st.code(NAVIGATOR_ENDPOINT)
        st.markdown("**Server:** http://127.0.0.1:8080")
        
        # Show status based on last source
        source = getattr(api_client, "last_source", "Not used yet")

        if "Mock" in source:
            st.warning(f"Using mock model fallback ({source})")
        elif "Navigator" in source:
            st.info(f"Using AI Navigator fallback ({source})")
        elif "Connect" in source:
            st.success(f"Using Production API ({source})")
        else:
            st.write(f"Status: {source}")
    
    with col2:
        source = getattr(api_client, "last_source", "Not used yet")

        st.markdown("**Model Information:**")
        if "Mock" in source:
            st.markdown("- **Type:** Mock Model (Demo)")
            st.markdown("- **Logic:** Heuristic-based")
            st.markdown("- **Purpose:** Demo when APIs down")
        elif "Navigator" in source:
            st.markdown("- **Type:** AI Navigator (Local)")
            st.markdown("- **Model:** Qwen 2.5 7B")
            st.markdown("- **Server:** Local (Port 8080)")
            st.markdown("- **Status:** Running")
        elif "Connect" in source:
            st.markdown("- **Type:** Deployed Model (Anaconda Connect)")
            st.markdown("- **Status:** Available")
        else:
            st.markdown("- **Status:** No inference run yet")

# ================================================================================
# FOOTER
# ================================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Powered by Anaconda Platform</strong></p>
    <p>Core - Desktop - AI Catalyst</p>
    <p>2026 Fraud Detection System | Built for Enterprise AI Deployments</p>
</div>
""", unsafe_allow_html=True)