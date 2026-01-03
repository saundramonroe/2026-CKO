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
# ================================================================================
# PAGE 3: ANALYTICS (DYNAMIC FRAUD PATTERNS)
# ================================================================================

elif page == "Analytics":
    st.title("Advanced Analytics & Fraud Intelligence")
    st.markdown("Monitor fraud patterns, trends, and system performance")
    
    # ============================================================================
    # TIME RANGE SELECTOR
    # ============================================================================
    
    col_selector, col_refresh = st.columns([4, 1])
    
    with col_selector:
        time_range = st.selectbox(
            "Select Time Range",
            ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Last 90 Days"],
            help="View fraud patterns and system metrics across different time periods"
        )
    
    with col_refresh:
        st.write("")  # Spacing
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
    
    st.markdown("---")
    
    # ============================================================================
    # FRAUD DISTRIBUTION BY CATEGORY (DYNAMIC)
    # ============================================================================
    
    st.subheader("📊 Fraud Distribution by Category")
    
    categories = ['Crypto/ATM', 'Online Shopping', 'Wire Transfers', 'Other', 'Gas Stations', 'Restaurants']
    
    # Different fraud patterns showing evolving trends over time
    fraud_data_by_range = {
        # Last 24 hours - Recent crypto spike
        "Last 24 Hours": {
            'counts': [8, 5, 4, 3, 2, 1],  # Total: 23 frauds
            'total_transactions': 12547,
            'note': '📈 Crypto fraud spike detected today',
            'insight': 'Crypto/ATM fraud at 34.8% - highest in 30 days',
            'recommendation': 'Consider additional verification for crypto transactions',
            'trend': 'up'
        },
        
        # Last 7 days - Crypto still elevated, shopping increasing
        "Last 7 Days": {
            'counts': [52, 38, 30, 22, 12, 7],  # Total: 161 frauds (~23/day)
            'total_transactions': 87829,
            'note': '⚠️ Crypto remains elevated this week',
            'insight': 'Online shopping fraud trending up (+15% vs previous week)',
            'recommendation': 'Monitor e-commerce transactions during sales events',
            'trend': 'stable'
        },
        
        # Last 30 days - Shopping fraud increasing (holiday season)
        "Last 30 Days": {
            'counts': [210, 165, 125, 95, 60, 35],  # Total: 690 frauds (~23/day)
            'total_transactions': 376320,
            'note': '🛍️ Shopping fraud trending up (holiday season)',
            'insight': 'E-commerce fraud up to 23.9% (from 18% baseline) - seasonal pattern',
            'recommendation': 'Enhanced monitoring for online retail during Q4',
            'trend': 'up'
        },
        
        # Last 90 days - Longer term balanced patterns
        "Last 90 Days": {
            'counts': [595, 525, 385, 280, 175, 110],  # Total: 2,070 frauds (~23/day)
            'total_transactions': 1128960,
            'note': '📊 Balanced fraud distribution over quarter',
            'insight': 'Crypto fraud decreased 6% quarter-over-quarter (better controls)',
            'recommendation': 'Maintain current crypto safeguards, monitor shopping trends',
            'trend': 'down'
        }
    }
    
    # Get data for selected range
    selected_data = fraud_data_by_range[time_range]
    fraud_counts = selected_data['counts']
    total_frauds = sum(fraud_counts)
    total_transactions = selected_data['total_transactions']
    fraud_rate = (total_frauds / total_transactions) * 100
    
    # Calculate percentages
    percentages = [(count/total_frauds)*100 for count in fraud_counts]
    
    # Create two columns for chart and stats
    col_chart, col_stats = st.columns([2, 1])
    
    with col_chart:
        # Create pie chart
        fig = px.pie(
            values=fraud_counts,
            names=categories,
            title=f"Fraud Cases by Category ({time_range})",
            color_discrete_sequence=px.colors.sequential.RdBu,
            hole=0.3  # Donut chart style
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>'
        )
        
        fig.update_layout(
            height=450,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col_stats:
        st.markdown("### 📈 Key Statistics")
        
        st.metric(
            "Total Fraud Cases",
            f"{total_frauds:,}",
            f"{time_range}"
        )
        
        st.metric(
            "Total Transactions",
            f"{total_transactions:,}",
            f"{fraud_rate:.3f}% fraud rate",
            delta_color="inverse"
        )
        
        highest_idx = fraud_counts.index(max(fraud_counts))
        st.metric(
            "Highest Risk",
            categories[highest_idx],
            f"{percentages[highest_idx]:.1f}% of fraud"
        )
        
        # Trend indicator
        trend_emoji = {
            'up': '📈 Increasing',
            'down': '📉 Decreasing', 
            'stable': '➡️ Stable'
        }
        
        st.markdown(f"**Trend:** {trend_emoji[selected_data['trend']]}")
    
    # Alert box with insight
    if selected_data['trend'] == 'up':
        st.warning(f"⚠️ **Alert:** {selected_data['note']}")
    elif selected_data['trend'] == 'down':
        st.success(f"✅ **Good News:** {selected_data['note']}")
    else:
        st.info(f"ℹ️ {selected_data['note']}")
    
    # Detailed insight
    st.markdown(f"**Analysis:** {selected_data['insight']}")
    st.markdown(f"**Recommendation:** {selected_data['recommendation']}")
    
    # Detailed breakdown table
    with st.expander("📊 View Detailed Category Breakdown"):
        breakdown_df = pd.DataFrame({
            'Category': categories,
            'Fraud Count': fraud_counts,
            'Percentage': [f"{p:.1f}%" for p in percentages],
            'Avg per Day': [f"{c / (1 if time_range == 'Last 24 Hours' else 7 if time_range == 'Last 7 Days' else 30 if time_range == 'Last 30 Days' else 90):.1f}" for c in fraud_counts]
        })
        
        # Add risk level
        def risk_level(pct):
            if pct > 30:
                return "🔴 Critical"
            elif pct > 20:
                return "🟡 High"
            elif pct > 10:
                return "🟢 Moderate"
            else:
                return "⚪ Low"
        
        breakdown_df['Risk Level'] = [risk_level(p) for p in percentages]
        
        st.dataframe(
            breakdown_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Category comparison
        st.markdown("### Category Insights")
        for i, cat in enumerate(categories):
            if percentages[i] > 25:
                st.markdown(f"- **{cat}**: Critical focus area ({percentages[i]:.1f}%) - {fraud_counts[i]:,} cases")
            elif percentages[i] > 15:
                st.markdown(f"- **{cat}**: High priority ({percentages[i]:.1f}%) - {fraud_counts[i]:,} cases")
    
    st.markdown("---")
    
    # ============================================================================
    # CATEGORY TREND OVER TIME (NEW SECTION)
    # ============================================================================
    
    st.subheader("📈 Category Trends Over Time")
    
    col_trend1, col_trend2 = st.columns(2)
    
    with col_trend1:
        # Show how top 3 categories have changed
        if time_range == "Last 90 Days":
            # Show weekly breakdown over 90 days
            weeks = ['Week 1-2', 'Week 3-4', 'Week 5-6', 'Week 7-8', 'Week 9-10', 'Week 11-12', 'Week 13']
            crypto_trend = [140, 135, 130, 125, 115, 105, 95]
            shopping_trend = [85, 90, 95, 105, 115, 125, 135]
            wire_trend = [65, 63, 62, 58, 55, 52, 50]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=weeks, y=crypto_trend, mode='lines+markers',
                                    name='Crypto/ATM', line=dict(color='#8B0000', width=3)))
            fig.add_trace(go.Scatter(x=weeks, y=shopping_trend, mode='lines+markers',
                                    name='Online Shopping', line=dict(color='#DC143C', width=3)))
            fig.add_trace(go.Scatter(x=weeks, y=wire_trend, mode='lines+markers',
                                    name='Wire Transfers', line=dict(color='#CD5C5C', width=3)))
            
            fig.update_layout(
                title="Top 3 Categories - 90 Day Trend",
                xaxis_title="Time Period",
                yaxis_title="Fraud Count",
                height=350,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.caption("🔍 Crypto fraud declining, shopping fraud increasing - possible shift in fraud tactics")
        
        else:
            # Show hourly/daily pattern for shorter ranges
            if time_range == "Last 24 Hours":
                x_points = list(range(24))
                x_label = "Hour of Day"
            elif time_range == "Last 7 Days":
                x_points = list(range(1, 8))
                x_label = "Day"
            else:  # 30 days
                x_points = list(range(1, 31))
                x_label = "Day"
            
            # Generate realistic patterns
            crypto_pattern = np.random.poisson(fraud_counts[0] / len(x_points), len(x_points))
            shopping_pattern = np.random.poisson(fraud_counts[1] / len(x_points), len(x_points))
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=x_points, y=crypto_pattern, name='Crypto/ATM', 
                               marker_color='#8B0000'))
            fig.add_trace(go.Bar(x=x_points, y=shopping_pattern, name='Online Shopping',
                               marker_color='#DC143C'))
            
            fig.update_layout(
                title=f"Fraud Distribution - {time_range}",
                xaxis_title=x_label,
                yaxis_title="Fraud Count",
                height=350,
                barmode='stack'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col_trend2:
        # Fraud rate by category
        st.markdown("#### Fraud Rate by Category")
        
        # Simulate detection rates by category
        detection_rates = {
            'Crypto/ATM': 0.92,
            'Online Shopping': 0.88,
            'Wire Transfers': 0.85,
            'Other': 0.87,
            'Gas Stations': 0.90,
            'Restaurants': 0.94
        }
        
        fig = go.Figure(go.Bar(
            y=list(detection_rates.keys()),
            x=list(detection_rates.values()),
            orientation='h',
            marker_color=['#00C851' if v > 0.90 else '#ffbb33' if v > 0.85 else '#ff4444' 
                         for v in detection_rates.values()],
            text=[f'{v:.1%}' for v in detection_rates.values()],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Detection Rate by Category",
            xaxis_title="Detection Rate",
            height=350,
            showlegend=False,
            xaxis=dict(range=[0.75, 1.0])
        )
        
        fig.add_vline(x=0.85, line_dash="dash", line_color="orange", 
                     annotation_text="Target: 85%")
        
        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 Higher detection rates = more fraud caught in that category")
    
    st.markdown("---")
    
    # ============================================================================
    # SYSTEM PERFORMANCE METRICS
    # ============================================================================
    
    st.subheader("⚙️ System Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⚡ Latency Trends")
        
        # Adjust number of days based on selection
        if time_range == "Last 24 Hours":
            x_data = list(range(24))
            x_label = "Hour"
            num_points = 24
            # Show some variation during business hours
            latency_data = []
            for hour in x_data:
                if 9 <= hour <= 17:  # Business hours - slightly higher
                    latency_data.append(48 + np.random.randn() * 5)
                else:
                    latency_data.append(42 + np.random.randn() * 4)
        elif time_range == "Last 7 Days":
            x_data = list(range(1, 8))
            x_label = "Day"
            num_points = 7
            latency_data = 45 + np.random.randn(7) * 4
        elif time_range == "Last 30 Days":
            x_data = list(range(1, 31))
            x_label = "Day"
            num_points = 30
            latency_data = 45 + np.random.randn(30) * 5
        else:  # Last 90 Days
            x_data = list(range(1, 91))
            x_label = "Day"
            num_points = 90
            # Show gradual improvement over 90 days
            latency_data = 50 - (np.arange(90) * 0.05) + np.random.randn(90) * 4
        
        # Ensure latency stays positive and realistic
        latency_data = np.clip(latency_data, 25, 80)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_data,
            y=latency_data,
            mode='lines+markers',
            name='Latency',
            line=dict(color='#667eea', width=2),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.1)',
            hovertemplate=f'<b>{x_label} %{{x}}</b><br>Latency: %{{y:.1f}}ms<extra></extra>'
        ))
        
        fig.add_hline(
            y=100,
            line_dash="dash",
            line_color="red",
            annotation_text="SLA: 100ms",
            annotation_position="right"
        )
        
        # Add average line
        avg_latency = np.mean(latency_data)
        fig.add_hline(
            y=avg_latency,
            line_dash="dot",
            line_color="green",
            annotation_text=f"Avg: {avg_latency:.1f}ms",
            annotation_position="left"
        )
        
        fig.update_layout(
            xaxis_title=x_label,
            yaxis_title="Latency (ms)",
            height=350,
            hovermode='x unified',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Latency statistics
        max_latency = np.max(latency_data)
        min_latency = np.min(latency_data)
        sla_violations = sum(1 for x in latency_data if x > 100)
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Avg", f"{avg_latency:.1f}ms")
        metric_col2.metric("Max", f"{max_latency:.1f}ms")
        metric_col3.metric("SLA Violations", sla_violations, delta_color="inverse")
        
        if avg_latency < 50:
            st.success("✅ Excellent performance - well below SLA")
        elif avg_latency < 75:
            st.info("ℹ️ Good performance - within acceptable range")
        elif avg_latency < 100:
            st.warning("⚠️ Approaching SLA limit - monitor closely")
        else:
            st.error("🚨 SLA violations detected - action required")
    
    with col2:
        st.markdown("#### 📈 Accuracy Trends")
        
        # Generate accuracy data matching time range
        if time_range == "Last 24 Hours":
            accuracy_data = 0.995 + np.random.randn(24) * 0.003
            x_data = list(range(24))
            x_label = "Hour"
        elif time_range == "Last 7 Days":
            accuracy_data = 0.996 + np.random.randn(7) * 0.002
            x_data = list(range(1, 8))
            x_label = "Day"
        elif time_range == "Last 30 Days":
            accuracy_data = 0.9955 + np.random.randn(30) * 0.003
            x_data = list(range(1, 31))
            x_label = "Day"
        else:  # Last 90 Days
            # Show slight improvement over time
            accuracy_data = 0.993 + (np.arange(90) * 0.00003) + np.random.randn(90) * 0.002
            x_data = list(range(1, 91))
            x_label = "Day"
        
        # Clip to realistic range
        accuracy_data = np.clip(accuracy_data, 0.980, 1.0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_data,
            y=accuracy_data,
            mode='lines+markers',
            name='Accuracy',
            line=dict(color='#00C851', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 200, 81, 0.1)',
            hovertemplate=f'<b>{x_label} %{{x}}</b><br>Accuracy: %{{y:.4f}}<extra></extra>'
        ))
        
        # Add target line
        fig.add_hline(
            y=0.985,
            line_dash="dash",
            line_color="orange",
            annotation_text="Target: 98.5%",
            annotation_position="right"
        )
        
        # Add average line
        avg_accuracy = np.mean(accuracy_data)
        fig.add_hline(
            y=avg_accuracy,
            line_dash="dot",
            line_color="darkgreen",
            annotation_text=f"Avg: {avg_accuracy:.3f}",
            annotation_position="left"
        )
        
        fig.update_layout(
            xaxis_title=x_label,
            yaxis_title="Accuracy",
            height=350,
            yaxis=dict(range=[0.97, 1.0]),
            hovermode='x unified',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Accuracy statistics
        min_accuracy = np.min(accuracy_data)
        below_target = sum(1 for x in accuracy_data if x < 0.985)
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Avg", f"{avg_accuracy:.4f}")
        metric_col2.metric("Min", f"{min_accuracy:.4f}")
        metric_col3.metric("Below Target", below_target, delta_color="inverse")
        
        if avg_accuracy >= 0.995:
            st.success("✅ Exceptional accuracy maintained")
        elif avg_accuracy >= 0.985:
            st.info("ℹ️ Meeting accuracy targets")
        else:
            st.warning("⚠️ Below target - consider model refresh")
    
    st.markdown("---")
    
    # ============================================================================
    # FRAUD DETECTION FUNNEL (NEW SECTION)
    # ============================================================================
    
    st.subheader("🔍 Fraud Detection Funnel")
    st.markdown(f"How transactions flow through the system ({time_range})")
    
    # Calculate funnel metrics
    total_txns = total_transactions
    flagged = int(total_txns * 0.02)  # 2% flagged for review
    confirmed_fraud = total_frauds
    false_positives = flagged - confirmed_fraud
    auto_approved = total_txns - flagged
    
    # Create funnel visualization
    col_funnel, col_funnel_stats = st.columns([2, 1])
    
    with col_funnel:
        fig = go.Figure(go.Funnel(
            y=['Total Transactions', 'Flagged for Review', 'Confirmed Fraud', 'Blocked'],
            x=[total_txns, flagged, confirmed_fraud, confirmed_fraud],
            textinfo="value+percent initial",
            marker=dict(color=['#667eea', '#ffbb33', '#ff4444', '#8B0000']),
            connector=dict(line=dict(color='gray', width=2))
        ))
        
        fig.update_layout(
            title="Transaction Processing Funnel",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col_funnel_stats:
        st.markdown("#### Funnel Breakdown")
        st.metric("Total Processed", f"{total_txns:,}")
        st.metric("Flagged", f"{flagged:,}", f"{(flagged/total_txns)*100:.2f}%")
        st.metric("True Fraud", f"{confirmed_fraud:,}", f"{(confirmed_fraud/flagged)*100:.1f}% precision")
        st.metric("False Positives", f"{false_positives:,}", f"{(false_positives/total_txns)*100:.3f}%")
        st.metric("Auto-Approved", f"{auto_approved:,}", f"{(auto_approved/total_txns)*100:.1f}%")
        
        st.markdown("---")
        st.markdown("**Efficiency Metrics:**")
        st.write(f"• Manual review rate: {(flagged/total_txns)*100:.2f}%")
        st.write(f"• Auto-approval rate: {(auto_approved/total_txns)*100:.1f}%")
        st.write(f"• Precision: {(confirmed_fraud/flagged)*100:.1f}%")
    
    st.markdown("---")
    
    # ============================================================================
    # COMPARATIVE ANALYSIS (NEW SECTION)
    # ============================================================================
    
    st.subheader("📊 Performance Comparison")
    
    col_comp1, col_comp2 = st.columns(2)
    
    with col_comp1:
        st.markdown("#### Model Performance vs Baseline")
        
        comparison_metrics = ['Fraud Detection', 'False Positives', 'Response Time', 'Manual Reviews']
        baseline_values = [78, 5.0, 85, 100]  # Baseline percentages/values
        current_values = [87.34, 0.47, 45, 60]  # Our model
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Baseline System',
            y=comparison_metrics,
            x=baseline_values,
            orientation='h',
            marker_color='#ff9999',
            text=[f'{v:.1f}' + ('%' if i < 2 else 'ms' if i == 2 else '%') 
                  for i, v in enumerate(baseline_values)],
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            name='Our Model',
            y=comparison_metrics,
            x=current_values,
            orientation='h',
            marker_color='#66b3ff',
            text=[f'{v:.1f}' + ('%' if i < 2 else 'ms' if i == 2 else '%') 
                  for i, v in enumerate(current_values)],
            textposition='auto'
        ))
        
        fig.update_layout(
            barmode='group',
            height=350,
            xaxis_title="Value",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("📈 Green bars show improvement areas")
    
    with col_comp2:
        st.markdown("#### Cost Savings Impact")
        
        # Calculate savings based on time range
        if time_range == "Last 24 Hours":
            period_multiplier = 1
            period_label = "Daily"
        elif time_range == "Last 7 Days":
            period_multiplier = 7
            period_label = "Weekly"
        elif time_range == "Last 30 Days":
            period_multiplier = 30
            period_label = "Monthly"
        else:
            period_multiplier = 90
            period_label = "Quarterly"
        
        # Calculate daily savings
        daily_fraud_prevented = 150 * 1.5  # ~1.5 additional frauds/day @ $150 each
        daily_fp_reduction = 200 * 0.75  # ~200 fewer FPs/day @ $75 cost savings each
        daily_total = daily_fraud_prevented + daily_fp_reduction
        
        period_savings = daily_total * period_multiplier
        
        savings_categories = ['Fraud Prevention', 'False Positive Reduction', 'Manual Review Savings']
        savings_values = [
            daily_fraud_prevented * period_multiplier,
            daily_fp_reduction * period_multiplier,
            100 * period_multiplier  # $100/day in manual review savings
        ]
        
        fig = go.Figure(go.Waterfall(
            name="Savings",
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=savings_categories + ["Total Savings"],
            textposition="outside",
            text=[f"${v:,.0f}" for v in savings_values] + [f"${sum(savings_values):,.0f}"],
            y=savings_values + [sum(savings_values)],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#00C851"}},
            increasing={"marker": {"color": "#00C851"}},
            totals={"marker": {"color": "#0066cc"}}
        ))
        
        fig.update_layout(
            title=f"{period_label} Cost Savings",
            height=350,
            showlegend=False,
            yaxis_title="Savings ($)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.metric(
            f"Total {period_label} Savings",
            f"${sum(savings_values):,.0f}",
            f"Annualized: ${sum(savings_values) * (365/period_multiplier):,.0f}"
        )
    
    st.markdown("---")
    
    # ============================================================================
    # ALERT SUMMARY (NEW SECTION)
    # ============================================================================
    
    st.subheader("🚨 Fraud Alert Summary")
    
    col_alert1, col_alert2, col_alert3, col_alert4 = st.columns(4)
    
    with col_alert1:
        st.metric(
            "High Risk Alerts",
            f"{int(total_frauds * 0.45)}",
            f"{((int(total_frauds * 0.45) / total_frauds) * 100):.1f}%"
        )
        st.caption("Score > 0.8")
    
    with col_alert2:
        st.metric(
            "Medium Risk Alerts", 
            f"{int(total_frauds * 0.35)}",
            f"{((int(total_frauds * 0.35) / total_frauds) * 100):.1f}%"
        )
        st.caption("Score 0.5-0.8")
    
    with col_alert3:
        st.metric(
            "Auto-Blocked",
            f"{int(total_frauds * 0.45)}",
            "100% fraud"
        )
        st.caption("Immediate action")
    
    with col_alert4:
        st.metric(
            "Manual Reviews",
            f"{int(total_frauds * 0.35)}",
            "-40% vs baseline",
            delta_color="inverse"
        )
        st.caption("Analyst workload")
    
    # Recent high-risk cases
    with st.expander("📋 View Recent High-Risk Cases"):
        # Generate sample high-risk cases
        num_samples = min(10, total_frauds)
        
        recent_cases = pd.DataFrame({
            'Timestamp': [(datetime.now() - timedelta(hours=i*2)).strftime('%Y-%m-%d %H:%M') 
                         for i in range(num_samples)],
            'Merchant': np.random.choice(
                ['BITCOIN ATM UNKNOWN', 'CRYPTO EXCHANGE UNVERIFIED', 'WIRE TRANSFER 9923',
                 'ONLINE CASINO DEPOSIT', 'FOREIGN CODE 5521', 'UNKNOWN MERCHANT 7734'],
                num_samples
            ),
            'Amount': [f"${x:,.2f}" for x in np.random.uniform(800, 5000, num_samples)],
            'Risk Score': [f"{x:.3f}" for x in np.random.uniform(0.85, 0.99, num_samples)],
            'Category': np.random.choice(['Crypto/ATM', 'Wire Transfers', 'Online Shopping'], num_samples),
            'Status': np.random.choice(['BLOCKED', 'UNDER REVIEW'], num_samples, p=[0.7, 0.3])
        })
        
        st.dataframe(recent_cases, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # ============================================================================
    # EXPORT DATA OPTION
    # ============================================================================
    
    col_export1, col_export2 = st.columns([3, 1])
    
    with col_export1:
        st.markdown("### 📥 Export Analytics Data")
        st.markdown("Download fraud analytics for further analysis or reporting")
    
    with col_export2:
        # Create export data
        export_df = pd.DataFrame({
            'Category': categories,
            'Fraud_Count': fraud_counts,
            'Percentage': [f"{p:.2f}" for p in percentages],
            'Time_Range': [time_range] * len(categories)
        })
        
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📊 Download CSV",
            data=csv,
            file_name=f"fraud_analytics_{time_range.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

        
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