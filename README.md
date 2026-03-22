# Reliance Stock Forecasting — Streamlit App

Attention-Augmented GRU forecasting app for NSE Reliance Industries.

## Setup

### 1. Install dependencies
```bash
pip install streamlit tensorflow scikit-learn joblib plotly pandas numpy
```

### 2. Folder structure
```
your_folder/
├── app.py
├── models/
│   ├── attn_gru_final.keras
│   ├── feature_scaler.pkl
│   ├── target_scaler.pkl
│   └── model_config.json
└── data/
    ├── RELIANCE.csv
    ├── error_metrics.csv
    ├── test_forecast.csv
    └── future_forecast_30days.csv
```

### 3. Run
```bash
streamlit run app.py
```

## Pages
- **Dashboard**       — KPIs, full price history, dataset summary
- **Live Prediction** — Next-day price prediction + N-day forecast
- **Backtest Analysis** — Test set actual vs predicted, metrics, scatter
- **Attention Insights** — Which past days the model focuses on
- **Future Forecast** — Autoregressive N-day ahead forecast with ±3% band
- **Model Info**      — Architecture, features, training config
