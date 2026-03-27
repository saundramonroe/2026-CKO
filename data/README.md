# Hybrid Fraud Detection System

> Showcasing Anaconda Core + Desktop + AI Catalyst

Enterprise-grade fraud detection combining traditional ML (XGBoost/Random Forest) with LLM intelligence (Meta-Llama-3.1-8B-Instruct via Anaconda Connect API). The two-stage architecture screens all transactions with fast ML, then calls the LLM only for high-risk cases — delivering sub-100ms latency at scale while maintaining explainability.

---

## Architecture

```
Transaction → [Stage 1: XGBoost/Random Forest] → Low risk? → APPROVE
                                                → High risk? → [Stage 2: LLM via Anaconda Connect API] → APPROVE / REVIEW / BLOCK
```

- **Stage 1 (XGBoost):** Screens 100% of transactions in <1ms each using 30 numeric features (PCA components, time, amount).
- **Stage 2 (LLM):** Analyzes merchant text for the ~1-3% of transactions that exceed the 0.3 risk threshold. Calls the Meta-Llama-3.1-8B-Instruct model deployed on Anaconda Connect — no local GPU or model download required.
- **Fallback chain:** Anaconda Connect → Anaconda Desktop (local) → Mock model (always available for demos).

---

## Quick Start

### 1. Set Up Environment

```bash
conda env create -f environment.yml
conda activate fraud-detection-qwen
```

The environment is lightweight (~800MB) since LLM inference runs via API rather than locally. No PyTorch, transformers, or GPU drivers needed.

### 2. Download Dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the `data/` directory:

```
data/
  creditcard.csv    ← 284,807 transactions, 492 fraud cases (0.172%)
```

### 3. Configure API Endpoint

The LLM endpoint is configured in `src/config.py`:

```python
CONNECT_ENDPOINT = "https://demo.se.sb.anacondaconnect.com/api/ai/inference/serve/.../v1/chat/completions"
API_TOKEN = "your-token-here"
```

If the Connect endpoint is unavailable, the system automatically falls back to a local Anaconda Desktop server or mock predictions.

### 4. Run the Notebooks

```bash
jupyter lab
```

Work through the notebooks in order:

| Notebook | Purpose | Runtime |
|---|---|---|
| `01_data_exploration.ipynb` | Dataset analysis, class imbalance visualization | ~2 min |
| `02_model_training.ipynb` | Train hybrid XGBoost + LLM detector | ~5-10 min (demo mode) |
| `03_evaluation.ipynb` | Performance metrics, confusion matrix, ROI analysis | ~3 min |
| `04_interactive_demo.ipynb` | Interactive widget-based transaction testing | Interactive |

### 5. Launch the Dashboard

```bash
streamlit run fraud-dashboard.py
```

The dashboard provides real-time fraud detection testing, analytics, and system health monitoring across four pages: Dashboard, Test Transaction, Analytics, and System Status.

---

## Project Structure

```
├── src/
│   ├── config.py          # Centralized configuration (endpoints, thresholds, model params)
│   ├── models.py          # Hybrid detector + LLM analysis via Anaconda Connect API
│   ├── data_utils.py      # Data loading, feature engineering, train/test splitting
│   └── api_client.py      # Multi-endpoint API client with fallback chain
│
├── 01_data_exploration.ipynb    # Dataset analysis & visualization
├── 02_model_training.ipynb      # Model training & evaluation
├── 03_evaluation.ipynb          # Detailed performance analysis
├── 04_interactive_demo.ipynb    # Interactive testing widgets
│
├── fraud-dashboard.py           # Streamlit dashboard (primary)
├── fraud-detection-dash.py      # Streamlit dashboard (alternate)
│
├── environment.yml              # Conda environment (lean, API-based)
├── data/                        # Dataset directory (not in repo)
└── models/                      # Saved model artifacts
```

---

## Environment

The `environment.yml` defines a lean conda environment with only the packages the project imports directly:

| Category | Packages |
|---|---|
| **Data Science** | numpy, pandas, scipy, scikit-learn |
| **ML Models** | xgboost, imbalanced-learn, shap, joblib |
| **Visualization** | matplotlib, seaborn, plotly, pillow |
| **Dashboard** | streamlit |
| **API / HTTP** | requests |
| **Jupyter** | jupyterlab, notebook, ipykernel, ipywidgets |

**Why no PyTorch or transformers?** The LLM (Meta-Llama-3.1-8B-Instruct) is deployed on Anaconda Connect and accessed via REST API. This eliminates ~4GB of local dependencies and removes the need for GPU drivers, CUDA, or Hugging Face authentication.

To recreate the environment from scratch:

```bash
conda env remove -n fraud-detection-qwen
conda env create -f environment.yml
conda activate fraud-detection-qwen
```

---

## Configuration

All settings live in `src/config.py`:

| Setting | Default | Description |
|---|---|---|
| `DEMO_MODE` | `True` | Fast demos (5-10 min) vs full analysis (60+ min) |
| `TRAIN_SAMPLE_SIZE` | 50,000 | Training set size in demo mode |
| `TEST_SAMPLE_SIZE` | 10,000 | Test set size in demo mode |
| `LLM_ANALYSIS_LIMIT` | 10 | Max LLM API calls during evaluation (demo mode) |
| `LOW_RISK_THRESHOLD` | 0.3 | XGBoost score that triggers LLM analysis |
| `HIGH_RISK_THRESHOLD` | 0.8 | Score above which transactions are auto-blocked |
| `MODEL_WEIGHTS` | xgb=0.6, llm=0.4 | Ensemble weighting between XGBoost and LLM |

Switch to full mode for production-grade results:

```python
DEMO_MODE = False  # Uses full 284K dataset, 100 LLM calls in evaluation
```

---

## Performance

| Metric | Score | Description |
|---|---|---|
| **Accuracy** | 99.94% | Overall correctness across all transactions |
| **Precision** | 73.33% | When we flag fraud, we're right 73% of the time |
| **Recall** | 84.62% | We catch 85 out of 100 actual frauds |
| **F1-Score** | 78.57% | Balanced precision/recall |
| **ROC-AUC** | 98.23% | Discrimination between fraud and legitimate |
| **Latency** | <50ms avg | XGBoost screening (LLM adds ~100-500ms for flagged cases) |
| **Throughput** | 5,000+ TPS | XGBoost-only screening capacity |

Compared to a 78% baseline detection rate, the hybrid model catches significantly more fraud while reducing false positives by ~40%.

---

## Key Changes (Recent)

### LLM Inference: Local → API

`src/models.py` was updated to route all LLM calls through the Anaconda Connect REST API instead of downloading the model locally via Hugging Face:

- **Removed:** `torch`, `transformers`, `AutoTokenizer`, `AutoModelForCausalLM`
- **Added:** `requests`-based HTTP calls to the `/v1/chat/completions` endpoint
- **Result:** No Hugging Face auth needed, no 4.7GB model download, no GPU required
- **Same interface:** `load_llm_model()`, `analyze_merchant_llm()`, `OptimizedHybridDetector` all work as before — notebooks don't need changes

### Dashboard Fix

`fraud-dashboard.py` had an undefined `check_system_health` function on the System Status page. Fix:

```python
# Line 2713: change check_system_health.clear() to:
st.cache_data.clear()

# Line 2715: change health_status = check_system_health() to:
health_status = api_client.test_connection()
```

---

## Anaconda Value Demonstrated

- **Core:** Package management, SBOM generation, dependency tracking, reproducible environments
- **Desktop:** Integrated Jupyter + ML environment, local LLM fallback via AI Navigator
- **AI Catalyst:** One-click model deployment, REST API generation, model governance, monitoring dashboards

---

## Related Resources

- [Anaconda Documentation](https://docs.anaconda.com)
- [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)