# Hybrid Fraud Detection System
> Showcasing Anaconda Core + Desktop + AI Catalyst

Enterprise-grade fraud detection combining traditional ML (XGBoost) with LLM intelligence (Qwen 2.5 7B).

## Quick Start

### 1. Set Up Environment (Anaconda Desktop)
```bash
conda env create -f environment.yml
conda activate fraud-detection-demo
```

### 2. Download Dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place in `data/` directory.

### 3. Run Interactive Dashboard
```bash
streamlit run app-fraud.py
```

Or explore the notebooks:
```bash
jupyter notebook
```

## Project Structure
- `src/` - Core Python modules (importable, reusable)
- `notebooks/` - Jupyter notebooks for exploration and training
- `app-fraud.py` - Streamlit dashboard for interactive demos
- `data/` - Dataset location (not included in repo)

## Anaconda Value Demonstrated
- **Core**: Package management, SBOM generation, dependency tracking
- **Desktop**: Integrated Jupyter + ML environment
- **AI Catalyst**: Model governance, deployment, monitoring

## Performance
- **Fraud Detection**: 87-92% (vs 78% baseline)
- **Response Time**: <50ms average
- **False Positives**: -53% reduction

## Related Resources
- [Anaconda Documentation](https://docs.anaconda.com)
