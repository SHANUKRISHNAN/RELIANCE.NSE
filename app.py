"""
Reliance Stock Forecasting — Attention-Augmented GRU
Streamlit Deployment App
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Reliance GRU Forecast",
    page_icon   = "📈",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; color: #e0e0e0; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d2e 0%, #0f1117 100%);
        border-right: 1px solid #2d2f3e;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2235 0%, #252840 100%);
        border: 1px solid #3d4270;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 6px 0;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #60a5fa;
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 4px;
    }
    .metric-card .sublabel {
        font-size: 0.75rem;
        color: #64748b;
    }

    /* Status badge */
    .badge-good  { background:#065f46; color:#6ee7b7;
                   padding:4px 12px; border-radius:20px; font-size:0.8rem; }
    .badge-warn  { background:#7c2d12; color:#fdba74;
                   padding:4px 12px; border-radius:20px; font-size:0.8rem; }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #93c5fd;
        border-bottom: 1px solid #2d4a7a;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }

    /* Info box */
    .info-box {
        background: #1e2a3a;
        border-left: 3px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 0.9rem;
        color: #cbd5e1;
    }

    /* Prediction highlight */
    .pred-highlight {
        background: linear-gradient(135deg, #0c4a6e, #1e3a5f);
        border: 1px solid #0ea5e9;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }
    .pred-highlight .price {
        font-size: 3rem;
        font-weight: 800;
        color: #38bdf8;
    }
    .pred-highlight .change-pos { color: #34d399; font-size: 1.2rem; }
    .pred-highlight .change-neg { color: #f87171; font-size: 1.2rem; }

    /* Hide streamlit branding */
    #MainMenu {visibility:hidden;}
    footer     {visibility:hidden;}
    header     {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
# ── Absolute paths ─────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = BASE_DIR  # files are in repo root
DATA_DIR   = BASE_DIR  # files are in repo root

FEATURE_COLS = [
    "Open","High","Low","Close","Volume",
    "Return","Log_Return",
    "HL_Ratio","OC_Ratio",
    "Momentum_5","Momentum_10",
    "Vol_10","MA_Ratio","Log_Volume"
]
SEQUENCE_LEN = 60


# ═══════════════════════════════════════════════════════════════════════════
# PURE NUMPY GRU + ATTENTION INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════
# No TensorFlow. No Keras. No version issues. Ever.
# Implements the exact same forward pass as the trained model.

def _sigmoid(x):   return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))
def _tanh(x):      return np.tanh(np.clip(x, -88, 88))
def _relu(x):      return np.maximum(0.0, x)
def _softmax(x):   e = np.exp(x - x.max(axis=-1, keepdims=True)); return e / e.sum(axis=-1, keepdims=True)


def _bn(x, gamma, beta, mean, var, eps=1e-3):
    """Batch Normalization forward pass (inference mode — uses stored mean/var)."""
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def _gru_step(x_t, h_prev, kernel, rec_kernel, bias):
    """
    Single GRU timestep (Keras default gate order: z, r, h).

    kernel      : (input_dim, 3*units)
    rec_kernel  : (units, 3*units)
    bias        : (2, 3*units)  → [input_bias, recurrent_bias]
    """
    units = h_prev.shape[-1]

    # Split kernels into gate slices: [z | r | h]
    Wz, Wr, Wh = kernel[:, :units],     kernel[:, units:2*units],     kernel[:, 2*units:]
    Uz, Ur, Uh = rec_kernel[:, :units], rec_kernel[:, units:2*units], rec_kernel[:, 2*units:]

    # Input and recurrent biases
    bz_i, br_i, bh_i = bias[0, :units], bias[0, units:2*units], bias[0, 2*units:]
    bz_r, br_r, bh_r = bias[1, :units], bias[1, units:2*units], bias[1, 2*units:]

    z = _sigmoid(x_t @ Wz + bz_i + h_prev @ Uz + bz_r)   # update gate
    r = _sigmoid(x_t @ Wr + br_i + h_prev @ Ur + br_r)   # reset gate
    h_cand = _tanh(x_t @ Wh + bh_i + r * (h_prev @ Uh + bh_r))  # candidate
    h_new  = (1.0 - z) * h_cand + z * h_prev
    return h_new


def _gru_layer(X, kernel, rec_kernel, bias):
    """
    Full GRU forward pass over sequence X.
    X shape: (batch, timesteps, input_dim)
    Returns all hidden states: (batch, timesteps, units)
    """
    batch, T, _  = X.shape
    units        = rec_kernel.shape[0]
    H            = np.zeros((batch, T, units), dtype=np.float32)
    h            = np.zeros((batch, units),    dtype=np.float32)

    for t in range(T):
        h      = _gru_step(X[:, t, :], h, kernel, rec_kernel, bias)
        H[:, t, :] = h

    return H


def _attention(H, W_kernel, W_bias, V_kernel, V_bias):
    """
    Bahdanau attention.
    H shape: (batch, T, units)
    Returns context (batch, units) and alpha weights (batch, T, 1)
    """
    score   = _tanh(H @ W_kernel + W_bias)             # (batch, T, attn_units)
    energy  = score @ V_kernel + V_bias                 # (batch, T, 1)
    alpha   = _softmax(energy)                          # (batch, T, 1)
    context = np.sum(alpha * H, axis=1)                 # (batch, units)
    return context, alpha


class NumpyGRU:
    """
    Pure NumPy forward-pass of AttnGRU_v4.
    Loaded from gru_weights.npz — no TensorFlow required.
    """
    def __init__(self, weights_path: str):
        w = np.load(weights_path, allow_pickle=False)

        # ── Print all keys for debugging ───────────────────────────────
        all_keys = list(w.files)

        def load(key):
            if key in w:
                return w[key].astype(np.float32)
            raise KeyError(f"Key '{key}' not found in npz. Available: {all_keys}")

        # ── GRU layers ─────────────────────────────────────────────────
        self.gru1_k  = load("gru1_kernel")
        self.gru1_rk = load("gru1_rec_kernel")
        self.gru1_b  = load("gru1_bias")

        self.gru2_k  = load("gru2_kernel")
        self.gru2_rk = load("gru2_rec_kernel")
        self.gru2_b  = load("gru2_bias")

        # ── Batch norms ────────────────────────────────────────────────
        self.bn1 = (load("bn1_gamma"), load("bn1_beta"),
                    load("bn1_mean"),  load("bn1_var"))
        self.bn2 = (load("bn2_gamma"), load("bn2_beta"),
                    load("bn2_mean"),  load("bn2_var"))

        # ── Attention ──────────────────────────────────────────────────
        self.attn_Wk = load("attn_W_kernel")
        self.attn_Wb = load("attn_W_bias")
        self.attn_Vk = load("attn_V_kernel")
        self.attn_Vb = load("attn_V_bias")

        # ── Dense head ─────────────────────────────────────────────────
        # Collect all dense layer kernels in sorted order
        # Keys look like: dense0_kernel, dense1_kernel, dense2_kernel
        # The LAST dense kernel with output dim=1 is the output layer
        dense_kernel_keys = sorted(
            [k for k in all_keys if "dense" in k and "kernel" in k
             and k != "out_kernel"],
            key=lambda k: k  # alphabetical: dense0 < dense1 < dense2
        )

        self.dense_layers = []
        for dk in dense_kernel_keys:
            bk = dk.replace("kernel", "bias")
            if bk in all_keys:
                W = w[dk].astype(np.float32)
                b = w[bk].astype(np.float32)
                # If output dim is 1 → this IS the output layer
                if W.shape[1] == 1:
                    self.out_k = W
                    self.out_b = b
                else:
                    self.dense_layers.append((W, b))

        # Explicit out_kernel/out_bias override if present
        if "out_kernel" in all_keys:
            self.out_k = load("out_kernel")
            self.out_b = load("out_bias")

        # ── Validate output layer exists ───────────────────────────────
        if not hasattr(self, "out_k"):
            # Last resort: use the last dense layer as output
            if self.dense_layers:
                self.out_k, self.out_b = self.dense_layers.pop()
            else:
                raise ValueError(
                    f"Cannot find output layer weights. "
                    f"Available keys: {all_keys}"
                )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        X shape: (batch, SEQUENCE_LEN, n_features)
        Returns: (batch, 1) — scaled log return predictions
        """
        X = X.astype(np.float32)

        # GRU 1 + BN 1
        H1 = _gru_layer(X,  self.gru1_k, self.gru1_rk, self.gru1_b)
        H1 = _bn(H1, *self.bn1)

        # GRU 2 + BN 2
        H2 = _gru_layer(H1, self.gru2_k, self.gru2_rk, self.gru2_b)
        H2 = _bn(H2, *self.bn2)

        # Attention
        context, self._last_alpha = _attention(
            H2, self.attn_Wk, self.attn_Wb, self.attn_Vk, self.attn_Vb
        )

        # Dense head
        x = context
        for i, (W, b) in enumerate(self.dense_layers):
            x = _relu(x @ W + b)

        # Output
        out = x @ self.out_k + self.out_b
        return out                               # (batch, 1)

    def get_attention_weights(self):
        """Returns last computed attention weights (batch, T, 1)."""
        return getattr(self, "_last_alpha", None)


# ═══════════════════════════════════════════════════════════════════════════
# LOAD ARTEFACTS  (cached)
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading model…")
def load_model_and_scalers():
    npz_path = os.path.join(MODELS_DIR, "gru_weights.npz")
    model    = NumpyGRU(npz_path)

    f_scaler = joblib.load(os.path.join(MODELS_DIR, "feature_scaler.pkl"))
    t_scaler = joblib.load(os.path.join(MODELS_DIR, "target_scaler.pkl"))

    with open(os.path.join(MODELS_DIR, "model_config.json")) as f:
        cfg = json.load(f)

    attn_model = None    # not needed — NumpyGRU stores alpha internally
    return model, attn_model, f_scaler, t_scaler, cfg


@st.cache_data(show_spinner="Reading saved results…")
def load_saved_results():
    metrics_df  = pd.read_csv(os.path.join(DATA_DIR, "error_metrics.csv"))
    forecast_df = pd.read_csv(os.path.join(DATA_DIR, "test_forecast.csv"),
                               parse_dates=["Date"])
    future_df   = pd.read_csv(os.path.join(DATA_DIR, "future_forecast_30days.csv"),
                               parse_dates=["Date"])
    return metrics_df, forecast_df, future_df


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Return"]      = df["Close"].pct_change()
    df["Log_Return"]  = np.log(df["Close"] / df["Close"].shift(1))
    df["HL_Ratio"]    = (df["High"] - df["Low"]) / df["Close"]
    df["OC_Ratio"]    = (df["Close"] - df["Open"]) / df["Open"]
    df["Momentum_5"]  = df["Close"] - df["Close"].shift(5)
    df["Momentum_10"] = df["Close"] - df["Close"].shift(10)
    df["Vol_10"]      = df["Return"].rolling(10).std()
    df["MA5"]         = df["Close"].rolling(5).mean()
    df["MA20"]        = df["Close"].rolling(20).mean()
    df["MA_Ratio"]    = df["MA5"] / df["MA20"]
    df.drop(columns=["MA5","MA20"], inplace=True)
    df["Log_Volume"]  = np.log(df["Volume"])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


def load_and_prepare(path_or_buffer):
    df = pd.read_csv(path_or_buffer)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    df = df[["Open","High","Low","Close","Volume"]]
    df = df.apply(pd.to_numeric, errors="coerce")
    df["Volume"] = df["Volume"].replace(0, np.nan)
    df.dropna(inplace=True)
    df = add_features(df)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# INFERENCE HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def make_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def reconstruct_price_rolling(lr_true, lr_pred, anchor_prices,
                               reset_every=20):
    prices_true = np.zeros(len(lr_true))
    prices_pred = np.zeros(len(lr_true))
    for i in range(len(lr_true)):
        if i % reset_every == 0:
            anchor = anchor_prices[i]
        start = i - (i % reset_every)
        prices_true[i] = anchor * np.exp(np.sum(lr_true[start:i+1]))
        prices_pred[i] = anchor * np.exp(np.sum(lr_pred[start:i+1]))
    return prices_true, prices_pred


def run_inference(df, model, attn_model, f_scaler, t_scaler, cfg):
    """Full inference pipeline on a prepared dataframe."""
    seq_len     = cfg["sequence_len"]
    reset_every = cfg.get("reset_every_days", 20)

    X_raw = f_scaler.transform(df[FEATURE_COLS])
    y_raw = t_scaler.transform(df[["Log_Return"]])
    X, y  = make_sequences(X_raw, y_raw, seq_len)

    y_pred_s  = model.predict(X)                         # (n, 1)
    attn_w    = model.get_attention_weights()            # (n, T, 1)
    y_pred_lr = t_scaler.inverse_transform(y_pred_s).flatten()
    y_true_lr = t_scaler.inverse_transform(y).flatten()

    dates   = df.index[seq_len:]
    anchors = df["Close"].iloc[seq_len:].values

    y_true_p, y_pred_p = reconstruct_price_rolling(
        y_true_lr, y_pred_lr, anchors, reset_every
    )

    return dates, y_true_p, y_pred_p, attn_w, y_true_lr, y_pred_lr


def predict_next_close(model, f_scaler, t_scaler, df, cfg):
    """Predict the next single day's close."""
    seq_len = cfg["sequence_len"]
    window  = df[FEATURE_COLS].iloc[-seq_len:]
    X       = f_scaler.transform(window)[np.newaxis].astype(np.float32)
    pred_s  = model.predict(X)                          # (1, 1)
    pred_lr = t_scaler.inverse_transform(pred_s)[0, 0]
    last    = float(df["Close"].iloc[-1])
    pred_p  = last * np.exp(pred_lr)
    return pred_p, pred_lr, last


def forecast_future(model, f_scaler, t_scaler, df, cfg, n_days=30):
    """Autoregressive n-day future forecast."""
    seq_len = cfg["sequence_len"]
    lr_idx  = FEATURE_COLS.index("Log_Return")
    last_p  = float(df["Close"].iloc[-1])

    X_raw   = f_scaler.transform(df[FEATURE_COLS])
    seq     = X_raw[-seq_len:][np.newaxis].astype(np.float32)

    log_rets = []
    for _ in range(n_days):
        pred_s  = model.predict(seq)                    # (1, 1)
        pred_lr = t_scaler.inverse_transform(pred_s)[0, 0]
        log_rets.append(pred_lr)
        new_row         = seq[0, -1, :].copy()
        new_row[lr_idx] = float(pred_s[0, 0])
        seq             = np.roll(seq, -1, axis=1)
        seq[0, -1, :]   = new_row

    prices = last_p * np.exp(np.cumsum(log_rets))
    return prices.tolist()


# ═══════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ═══════════════════════════════════════════════════════════════════════════
DARK_LAYOUT = dict(
    paper_bgcolor = "#0f1117",
    plot_bgcolor  = "#0f1117",
    font          = dict(color="#e0e0e0", size=12),
    xaxis         = dict(gridcolor="#1e2235", zeroline=False),
    yaxis         = dict(gridcolor="#1e2235", zeroline=False),
    legend        = dict(bgcolor="#1a1d2e", bordercolor="#3d4270",
                         borderwidth=1),
    margin        = dict(l=20, r=20, t=40, b=20),
)


def chart_price_history(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        name="Close", line=dict(color="#60a5fa", width=1.2)
    ))
    ma20 = df["Close"].rolling(20).mean()
    ma50 = df["Close"].rolling(50).mean()
    fig.add_trace(go.Scatter(
        x=df.index, y=ma20, name="MA-20",
        line=dict(color="#f59e0b", width=1, dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=ma50, name="MA-50",
        line=dict(color="#a78bfa", width=1, dash="dot")
    ))
    fig.update_layout(title="Closing Price — Full History",
                      yaxis_title="₹ Price", **DARK_LAYOUT)
    return fig


def chart_actual_vs_pred(dates, y_true, y_pred, title="Actual vs Predicted"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=y_true, name="Actual",
        line=dict(color="#60a5fa", width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=y_pred, name="Predicted",
        line=dict(color="#f87171", width=1.5, dash="dash")
    ))
    fig.update_layout(title=title, yaxis_title="₹ Price", **DARK_LAYOUT)
    return fig


def chart_residuals(dates, residuals):
    colors = ["#34d399" if r >= 0 else "#f87171" for r in residuals]
    fig = go.Figure(go.Bar(
        x=dates, y=residuals, marker_color=colors, name="Residual"
    ))
    fig.add_hline(y=0, line_color="white", line_width=0.8)
    fig.update_layout(title="Residuals (Actual − Predicted)",
                      yaxis_title="₹ Error", **DARK_LAYOUT)
    return fig


def chart_attention(attn_weights, seq_len=60):
    avg  = attn_weights.mean(axis=0).flatten()
    days = list(range(seq_len, 0, -1))

    top5 = set(np.argsort(avg)[-5:])
    colors = ["#ef4444" if i in top5 else "#60a5fa"
              for i in range(len(avg))]

    fig = go.Figure(go.Bar(
        x=days, y=avg, marker_color=colors,
        hovertemplate="Days ago: %{x}<br>Weight: %{y:.4f}<extra></extra>"
    ))
    fig.update_layout(
        title="Attention Focus — Which Past Days Matter Most?<br>"
              "<sub style='color:#94a3b8'>Red = Top 5 attended days</sub>",
        xaxis_title="Days Ago",
        yaxis_title="Avg Attention Weight",
        **DARK_LAYOUT
    )
    return fig


def chart_future_forecast(hist_dates, hist_true, hist_pred,
                           future_dates, future_preds, last_known):
    fig = go.Figure()
    # Historical actual
    fig.add_trace(go.Scatter(
        x=hist_dates, y=hist_true, name="Historical Actual",
        line=dict(color="#60a5fa", width=1.5)
    ))
    # Model fit on test
    fig.add_trace(go.Scatter(
        x=hist_dates, y=hist_pred, name="Model (test)",
        line=dict(color="#f87171", width=1.2, dash="dash")
    ))
    # Future forecast with confidence shading
    upper = [p * 1.03 for p in future_preds]
    lower = [p * 0.97 for p in future_preds]

    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=upper + lower[::-1],
        fill="toself", fillcolor="rgba(52,211,153,0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        name="±3% band", showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=future_preds, name="30-Day Forecast",
        line=dict(color="#34d399", width=2),
        mode="lines+markers",
        marker=dict(size=5, color="#34d399")
    ))
    # Vertical divider
    fig.add_vline(
        x=hist_dates[-1], line_color="#94a3b8",
        line_dash="dot", line_width=1
    )
    fig.add_annotation(
        x=hist_dates[-1], y=last_known,
        text="Forecast Start", showarrow=True,
        arrowcolor="#94a3b8", font=dict(color="#94a3b8", size=11)
    )
    fig.update_layout(
        title="30-Day Future Price Forecast with ±3% Confidence Band",
        yaxis_title="₹ Price", **DARK_LAYOUT
    )
    return fig


def chart_fold_metrics(metrics_df):
    folds  = metrics_df["Fold"].astype(str).tolist()
    mapes  = metrics_df["MAPE %"].tolist()
    r2s    = metrics_df["R²"].tolist()
    colors = ["#ef4444" if f == "TEST" else "#60a5fa" for f in folds]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["MAPE % per Fold", "R² per Fold"])
    fig.add_trace(go.Bar(x=folds, y=mapes, marker_color=colors,
                         name="MAPE %",
                         text=[f"{v:.2f}%" for v in mapes],
                         textposition="outside"), row=1, col=1)
    fig.add_trace(go.Bar(x=folds, y=r2s, marker_color=colors,
                         name="R²",
                         text=[f"{v:.4f}" for v in r2s],
                         textposition="outside"), row=1, col=2)
    fig.add_hline(y=5, line_dash="dot", line_color="#f59e0b",
                  annotation_text="5% threshold", row=1, col=1)
    fig.update_layout(showlegend=False,
                      paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                      font=dict(color="#e0e0e0"),
                      margin=dict(l=20, r=20, t=60, b=20))
    fig.update_xaxes(gridcolor="#1e2235")
    fig.update_yaxes(gridcolor="#1e2235")
    return fig


def chart_scatter(y_true, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred, mode="markers",
        marker=dict(color="#60a5fa", size=4, opacity=0.5),
        name="Predictions"
    ))
    lim = [min(y_true.min(), y_pred.min()) * 0.97,
           max(y_true.max(), y_pred.max()) * 1.03]
    fig.add_trace(go.Scatter(
        x=lim, y=lim, mode="lines",
        line=dict(color="#f87171", width=1.5, dash="dash"),
        name="Perfect prediction"
    ))
    fig.update_layout(
        title="Scatter: Actual vs Predicted — Test Set",
        xaxis_title="Actual ₹", yaxis_title="Predicted ₹",
        **DARK_LAYOUT
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0 8px'>
        <div style='font-size:2.5rem'>📈</div>
        <div style='font-size:1.2rem; font-weight:700; color:#60a5fa'>
            Reliance GRU
        </div>
        <div style='font-size:0.8rem; color:#64748b'>
            Attention-Augmented Forecasting
        </div>
    </div>
    <hr style='border-color:#2d2f3e; margin:12px 0'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠  Dashboard",
         "🔮  Live Prediction",
         "📊  Backtest Analysis",
         "🧠  Attention Insights",
         "📅  Future Forecast",
         "⚙️  Model Info"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#2d2f3e; margin:12px 0'>",
                unsafe_allow_html=True)

    # CSV uploader
    st.markdown(
        "<div style='font-size:0.85rem; color:#94a3b8; margin-bottom:8px'>"
        "📂 Upload your stock CSV</div>",
        unsafe_allow_html=True
    )
    uploaded = st.file_uploader(
        "CSV Upload", type=["csv"], label_visibility="collapsed"
    )

    # Forecast horizon slider (used in Live Prediction page)
    st.markdown(
        "<div style='font-size:0.85rem; color:#94a3b8;"
        " margin:12px 0 4px'>📆 Forecast horizon</div>",
        unsafe_allow_html=True
    )
    n_future = st.slider("Days", 5, 60, 30, label_visibility="collapsed")

    st.markdown(
        "<div style='font-size:0.75rem; color:#475569; margin-top:16px'>"
        "Model: AttnGRU-v4 · GRU-256/128<br>"
        "Data: NSE Reliance (1996–2026)<br>"
        "Test MAPE: 3.33% · R²: 0.87"
        "</div>",
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════════════════════════
# LOAD EVERYTHING
# ═══════════════════════════════════════════════════════════════════════════
model, attn_model, f_scaler, t_scaler, cfg = load_model_and_scalers()
metrics_df, saved_forecast_df, saved_future_df = load_saved_results()

# Load data — prefer uploaded, else default
@st.cache_data(show_spinner="Preparing data…")
def get_df(buf):
    src = buf if buf is not None else os.path.join(DATA_DIR, "RELIANCE.csv")
    return load_and_prepare(src)

df = get_df(uploaded)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠  Dashboard":
    st.markdown("## 📈 Reliance Stock Forecasting Dashboard")
    st.markdown(
        "<div class='info-box'>Attention-Augmented GRU trained with "
        "walk-forward validation on NSE Reliance Industries data (1996–2026). "
        "Uses log-return prediction with rolling-anchor ₹ reconstruction.</div>",
        unsafe_allow_html=True
    )

    # ── KPI row ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    last_close = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2])
    chg        = last_close - prev_close
    chg_pct    = chg / prev_close * 100

    with c1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='value'>₹{last_close:,.1f}</div>
            <div class='label'>Last Close</div>
            <div class='sublabel'>{df.index[-1].date()}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        arrow = "▲" if chg >= 0 else "▼"
        col   = "#34d399" if chg >= 0 else "#f87171"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='value' style='color:{col}'>{arrow} ₹{abs(chg):.1f}</div>
            <div class='label'>Day Change</div>
            <div class='sublabel'>{chg_pct:+.2f}%</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        test_row  = metrics_df[metrics_df["Fold"] == "TEST"].iloc[0]
        st.markdown(f"""
        <div class='metric-card'>
            <div class='value'>{test_row['MAPE %']:.2f}%</div>
            <div class='label'>Test MAPE</div>
            <div class='sublabel'>Rolling anchor reset=20d</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='value'>{test_row['R²']:.4f}</div>
            <div class='label'>Test R²</div>
            <div class='sublabel'>Variance explained</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Full price history ────────────────────────────────────────────────
    st.plotly_chart(chart_price_history(df), use_container_width=True)

    # ── Data summary ──────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-header'>Dataset Summary</div>",
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Metric": ["Total rows", "Start date", "End date",
                       "Min Close", "Max Close", "Avg Volume"],
            "Value": [
                f"{len(df):,}",
                str(df.index[0].date()),
                str(df.index[-1].date()),
                f"₹{df['Close'].min():.2f}",
                f"₹{df['Close'].max():.2f}",
                f"{df['Volume'].mean():,.0f}"
            ]
        }), use_container_width=True, hide_index=True)

    with col2:
        st.markdown("<div class='section-header'>Walk-Forward Fold Results</div>",
                    unsafe_allow_html=True)
        display_df = metrics_df.copy()
        display_df["MAPE %"] = display_df["MAPE %"].map("{:.3f}%".format)
        display_df["R²"]     = display_df["R²"].map("{:.4f}".format)
        st.dataframe(display_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: LIVE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔮  Live Prediction":
    st.markdown("## 🔮 Live Price Prediction")

    if len(df) < SEQUENCE_LEN + 10:
        st.error(f"Need at least {SEQUENCE_LEN + 10} rows of data.")
        st.stop()

    # ── Next-day prediction ───────────────────────────────────────────────
    pred_price, pred_lr, last_p = predict_next_close(
        model, f_scaler, t_scaler, df, cfg
    )
    chg     = pred_price - last_p
    chg_pct = chg / last_p * 100
    arrow   = "▲" if chg >= 0 else "▼"
    chg_cls = "change-pos" if chg >= 0 else "change-neg"

    st.markdown(f"""
    <div class='pred-highlight'>
        <div style='color:#94a3b8; font-size:0.9rem; margin-bottom:8px'>
            Next Trading Day Predicted Close
        </div>
        <div class='price'>₹{pred_price:,.2f}</div>
        <div class='{chg_cls}' style='margin-top:8px'>
            {arrow} ₹{abs(chg):.2f} &nbsp;({chg_pct:+.2f}%)
        </div>
        <div style='color:#64748b; font-size:0.8rem; margin-top:8px'>
            Based on last {SEQUENCE_LEN} trading days &nbsp;·&nbsp;
            Last known: ₹{last_p:,.2f} on {df.index[-1].date()}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Future forecast ───────────────────────────────────────────────────
    st.markdown(f"<div class='section-header'>{n_future}-Day Forecast</div>",
                unsafe_allow_html=True)

    with st.spinner(f"Running {n_future}-day autoregressive forecast…"):
        future_prices = forecast_future(model, f_scaler, t_scaler,
                                        df, cfg, n_days=n_future)

    future_dates = pd.bdate_range(
        start   = df.index[-1] + pd.Timedelta(days=1),
        periods = n_future
    )

    # Show last 90 days of test for context
    hist_n  = min(90, len(saved_forecast_df))
    hist_df = saved_forecast_df.tail(hist_n)

    fig = chart_future_forecast(
        hist_dates   = hist_df["Date"].values,
        hist_true    = hist_df["Actual_Close"].values,
        hist_pred    = hist_df["Predicted_Close"].values,
        future_dates = future_dates,
        future_preds = future_prices,
        last_known   = last_p
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Forecast table ────────────────────────────────────────────────────
    with st.expander("📋 View forecast table"):
        forecast_tbl = pd.DataFrame({
            "Date"            : future_dates.strftime("%Y-%m-%d"),
            "Predicted Close" : [f"₹{p:,.2f}" for p in future_prices],
            "Change from Now" : [f"₹{p-last_p:+.2f}  ({(p-last_p)/last_p*100:+.2f}%)"
                                  for p in future_prices]
        })
        st.dataframe(forecast_tbl, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: BACKTEST ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊  Backtest Analysis":
    st.markdown("## 📊 Backtest Analysis — Test Set (Unseen)")

    st.markdown(
        "<div class='info-box'>Results on the held-out test set "
        "(2021-06 → 2026-03). The model never saw this data during training. "
        "Prices reconstructed via rolling anchor (reset every 20 days).</div>",
        unsafe_allow_html=True
    )

    # ── Metrics ──────────────────────────────────────────────────────────
    test_row = metrics_df[metrics_df["Fold"] == "TEST"].iloc[0]
    c1, c2, c3, c4 = st.columns(4)

    def mcard(col, val, label, sub=""):
        col.markdown(f"""
        <div class='metric-card'>
            <div class='value'>{val}</div>
            <div class='label'>{label}</div>
            <div class='sublabel'>{sub}</div>
        </div>""", unsafe_allow_html=True)

    # Derive MAE/RMSE from saved forecast
    mae  = (saved_forecast_df["Actual_Close"] -
            saved_forecast_df["Predicted_Close"]).abs().mean()
    rmse = np.sqrt(((saved_forecast_df["Actual_Close"] -
                     saved_forecast_df["Predicted_Close"])**2).mean())

    mcard(c1, f"₹{mae:.2f}",          "MAE",    "Mean absolute ₹ error")
    mcard(c2, f"₹{rmse:.2f}",         "RMSE",   "Penalizes outlier days")
    mcard(c3, f"{test_row['MAPE %']:.3f}%", "MAPE", "Rolling anchor reset=20d")
    mcard(c4, f"{test_row['R²']:.4f}", "R²",    "Variance explained")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main forecast chart ───────────────────────────────────────────────
    st.plotly_chart(
        chart_actual_vs_pred(
            saved_forecast_df["Date"],
            saved_forecast_df["Actual_Close"],
            saved_forecast_df["Predicted_Close"],
            "Actual vs Predicted — Unseen Test Set"
        ),
        use_container_width=True
    )

    col1, col2 = st.columns(2)
    with col1:
        # Residuals bar
        st.plotly_chart(
            chart_residuals(
                saved_forecast_df["Date"],
                saved_forecast_df["Residual"]
            ),
            use_container_width=True
        )
    with col2:
        # Scatter
        st.plotly_chart(
            chart_scatter(
                saved_forecast_df["Actual_Close"].values,
                saved_forecast_df["Predicted_Close"].values
            ),
            use_container_width=True
        )

    # ── Fold metrics comparison ───────────────────────────────────────────
    st.markdown("<div class='section-header'>Walk-Forward Fold Metrics</div>",
                unsafe_allow_html=True)
    st.plotly_chart(chart_fold_metrics(metrics_df), use_container_width=True)

    # ── Raw data ──────────────────────────────────────────────────────────
    with st.expander("📋 View raw forecast data"):
        st.dataframe(
            saved_forecast_df.assign(
                **{c: saved_forecast_df[c].map("₹{:,.2f}".format)
                   for c in ["Actual_Close","Predicted_Close","Residual"]}
            ),
            use_container_width=True, hide_index=True
        )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: ATTENTION INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🧠  Attention Insights":
    st.markdown("## 🧠 Attention Weight Insights")

    st.markdown(
        "<div class='info-box'>Attention weights reveal which of the last "
        "60 trading days the model focuses on most. Computed live using "
        "the NumPy GRU engine. Red bars = top 5 most attended days.</div>",
        unsafe_allow_html=True
    )

    if len(df) < SEQUENCE_LEN + 5:
        st.error("Not enough data rows.")
        st.stop()

    with st.spinner("Computing attention weights…"):
        subset = df.tail(200 + SEQUENCE_LEN)
        X_raw  = f_scaler.transform(subset[FEATURE_COLS])
        y_raw  = t_scaler.transform(subset[["Log_Return"]])
        X, _   = make_sequences(X_raw, y_raw, SEQUENCE_LEN)
        _      = model.predict(X)                        # populates internal alpha
        attn_w = model.get_attention_weights()           # (n, T, 1)

    tab1, tab2 = st.tabs(["📊 Average (Last 200 days)", "🔍 Single day"])

    with tab1:
        st.plotly_chart(
            chart_attention(attn_w, SEQUENCE_LEN),
            use_container_width=True
        )
        avg      = attn_w.mean(axis=0).flatten()
        top5     = sorted(np.argsort(avg)[-5:][::-1],
                          key=lambda i: avg[i], reverse=True)
        days_ago = list(range(SEQUENCE_LEN, 0, -1))

        st.markdown("<div class='section-header'>Top 5 Attended Days</div>",
                    unsafe_allow_html=True)
        cols = st.columns(5)
        for col, idx in zip(cols, top5):
            col.markdown(f"""
            <div class='metric-card'>
                <div class='value' style='color:#ef4444'>
                    {days_ago[idx]}d
                </div>
                <div class='label'>Days ago</div>
                <div class='sublabel'>Weight: {avg[idx]:.4f}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(
            "<div class='info-box'><b>How to read this:</b> If the highest "
            "bars are near the right (days_ago = 1–5), the model relies on "
            "recent momentum. Spikes further left suggest recurring pattern "
            "at that lag distance (e.g. weekly cycles, earnings windows)."
            "</div>",
            unsafe_allow_html=True
        )

    with tab2:
        n_samples = len(attn_w)
        idx = st.slider("Sample index", 0, n_samples - 1, n_samples - 1)
        st.plotly_chart(
            chart_attention(attn_w[idx:idx+1], SEQUENCE_LEN),
            use_container_width=True
        )
        st.caption(f"Date: {subset.index[SEQUENCE_LEN + idx].date()}")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: FUTURE FORECAST
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📅  Future Forecast":
    st.markdown("## 📅 Future Price Forecast")

    st.markdown(
        "<div class='info-box'>Autoregressive forecast: each predicted "
        "log return is fed back as input for the next step. Use the slider "
        "in the sidebar to control the forecast horizon.</div>",
        unsafe_allow_html=True
    )

    if len(df) < SEQUENCE_LEN + 5:
        st.error("Not enough data.")
        st.stop()

    last_p = float(df["Close"].iloc[-1])

    with st.spinner(f"Generating {n_future}-day forecast…"):
        future_prices = forecast_future(model, f_scaler, t_scaler,
                                        df, cfg, n_days=n_future)

    future_dates = pd.bdate_range(
        start   = df.index[-1] + pd.Timedelta(days=1),
        periods = n_future
    )

    # ── KPIs ─────────────────────────────────────────────────────────────
    end_p    = future_prices[-1]
    total_ch = end_p - last_p
    total_pct= total_ch / last_p * 100
    peak     = max(future_prices)
    trough   = min(future_prices)

    c1, c2, c3, c4 = st.columns(4)
    col_ch = "#34d399" if total_ch >= 0 else "#f87171"
    with c1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='value'>₹{last_p:,.1f}</div>
            <div class='label'>Current Price</div>
            <div class='sublabel'>{df.index[-1].date()}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='value' style='color:{col_ch}'>
                ₹{end_p:,.1f}
            </div>
            <div class='label'>Day {n_future} Forecast</div>
            <div class='sublabel'>{future_dates[-1].date()}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='value' style='color:{col_ch}'>
                {total_pct:+.2f}%
            </div>
            <div class='label'>Total Change</div>
            <div class='sublabel'>Over {n_future} trading days</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='value'>₹{peak-trough:,.1f}</div>
            <div class='label'>Forecast Range</div>
            <div class='sublabel'>₹{trough:,.0f} – ₹{peak:,.0f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chart ─────────────────────────────────────────────────────────────
    hist_n  = min(90, len(saved_forecast_df))
    hist_df = saved_forecast_df.tail(hist_n)

    fig = chart_future_forecast(
        hist_dates   = hist_df["Date"].values,
        hist_true    = hist_df["Actual_Close"].values,
        hist_pred    = hist_df["Predicted_Close"].values,
        future_dates = future_dates,
        future_preds = future_prices,
        last_known   = last_p
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Saved 30-day results ──────────────────────────────────────────────
    st.markdown("<div class='section-header'>Saved 30-Day Forecast (from training)</div>",
                unsafe_allow_html=True)
    st.plotly_chart(
        go.Figure(
            go.Scatter(
                x=saved_future_df["Date"],
                y=saved_future_df["Predicted_Close"],
                mode="lines+markers",
                line=dict(color="#34d399", width=2),
                marker=dict(size=5)
            )
        ).update_layout(
            title="Saved 30-Day Forecast (generated at training time)",
            yaxis_title="₹ Price", **DARK_LAYOUT
        ),
        use_container_width=True
    )

    # ── Table ──────────────────────────────────────────────────────────────
    with st.expander("📋 View forecast table"):
        tbl = pd.DataFrame({
            "Date"             : future_dates.strftime("%Y-%m-%d"),
            "Predicted Close"  : [f"₹{p:,.2f}" for p in future_prices],
            "Change from Now"  : [
                f"₹{p-last_p:+,.2f}  ({(p-last_p)/last_p*100:+.2f}%)"
                for p in future_prices
            ]
        })
        st.dataframe(tbl, use_container_width=True, hide_index=True)

        csv = pd.DataFrame({
            "Date": future_dates.strftime("%Y-%m-%d"),
            "Predicted_Close": future_prices
        }).to_csv(index=False)
        st.download_button(
            "⬇️ Download forecast CSV", csv,
            file_name="reliance_forecast.csv", mime="text/csv"
        )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: MODEL INFO
# ═══════════════════════════════════════════════════════════════════════════
elif page == "⚙️  Model Info":
    st.markdown("## ⚙️ Model Architecture & Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-header'>Architecture</div>",
                    unsafe_allow_html=True)

        arch_rows = [
            ("Input",             f"(None, {cfg['sequence_len']}, {len(FEATURE_COLS)})"),
            ("GRU Layer 1",       f"units={cfg['gru_units_1']}, return_seq=True"),
            ("BatchNorm 1",       "—"),
            ("GRU Layer 2",       f"units={cfg['gru_units_2']}, return_seq=True"),
            ("BatchNorm 2",       "—"),
            ("BahdanauAttention", f"units={cfg['attn_units']}"),
            ("Dense 1",           "64, relu + L2"),
            ("Dropout",           f"{cfg['dropout']}"),
            ("Dense 2",           "32, relu + L2"),
            ("Output",            "Dense(1) → log_return"),
        ]
        st.dataframe(
            pd.DataFrame(arch_rows, columns=["Layer", "Config"]),
            use_container_width=True, hide_index=True
        )

        st.markdown("<div class='section-header' style='margin-top:16px'>"
                    "Training Config</div>",
                    unsafe_allow_html=True)
        train_rows = [
            ("Loss",              "Huber"),
            ("Optimizer",         "Adam"),
            ("Learning rate",     "1e-4 → 1e-3 (warmup 10 epochs)"),
            ("Gradient clip",     str(cfg["gradient_clip"])),
            ("Dropout",           str(cfg["dropout"])),
            ("L2 regularization", str(cfg["l2_reg"])),
            ("Max epochs",        "300"),
            ("EarlyStopping patience", "25"),
            ("ReduceLROnPlateau patience", "15"),
            ("Batch size",        "32"),
            ("Validation strategy", f"Walk-forward ({cfg['n_folds']} folds)"),
        ]
        st.dataframe(
            pd.DataFrame(train_rows, columns=["Parameter", "Value"]),
            use_container_width=True, hide_index=True
        )

    with col2:
        st.markdown("<div class='section-header'>Features (14 total)</div>",
                    unsafe_allow_html=True)
        feat_rows = [
            ("Open, High, Low, Close, Volume", "Raw OHLCV"),
            ("Return",       "Daily % change"),
            ("Log_Return",   "Log(Close_t / Close_{t-1}) — TARGET"),
            ("HL_Ratio",     "(High−Low) / Close — intraday volatility"),
            ("OC_Ratio",     "(Close−Open) / Open — candle body"),
            ("Momentum_5",   "Close − Close.shift(5)"),
            ("Momentum_10",  "Close − Close.shift(10)"),
            ("Vol_10",       "10-day rolling std of Return"),
            ("MA_Ratio",     "MA5 / MA20 — trend signal"),
            ("Log_Volume",   "log(Volume) — compressed tail"),
        ]
        st.dataframe(
            pd.DataFrame(feat_rows, columns=["Feature", "Description"]),
            use_container_width=True, hide_index=True
        )

        st.markdown("<div class='section-header' style='margin-top:16px'>"
                    "Key Design Decisions</div>",
                    unsafe_allow_html=True)
        decisions = [
            ("Target", "Log_Return (not raw Close) — stationary, no OOD explosion"),
            ("Scaler",  "StandardScaler (not MinMaxScaler) — handles unseen price ranges"),
            ("Evaluation", "Rolling anchor reset=20d — prevents cumsum error compounding"),
            ("Validation", "Walk-forward 4-fold — exposes model to all market regimes"),
            ("Reconstruction", "price_t = anchor × exp(Σ log_returns)"),
            ("LR MAPE", "Removed — near-zero denominator gives misleading 100%+ values"),
        ]
        st.dataframe(
            pd.DataFrame(decisions, columns=["Decision", "Reason"]),
            use_container_width=True, hide_index=True
        )

        st.markdown("<div class='section-header' style='margin-top:16px'>"
                    "Model Config JSON</div>",
                    unsafe_allow_html=True)
        st.json(cfg)
