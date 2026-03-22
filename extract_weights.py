"""
extract_weights.py
==================
Run this script ONCE on your local machine (where the .keras model works).
It extracts all GRU + Attention + Dense weights into a single .npz file
that is completely TensorFlow-version-independent.

Usage:
    python extract_weights.py

Output:
    models/gru_weights.npz   ← upload this to your GitHub repo
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

# ── Register custom layer ──────────────────────────────────────────────────
class BahdanauAttention(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W     = layers.Dense(units)
        self.V     = layers.Dense(1)

    def call(self, hidden_states):
        score   = tf.nn.tanh(self.W(hidden_states))
        alpha   = tf.nn.softmax(self.V(score), axis=1)
        context = tf.reduce_sum(alpha * hidden_states, axis=1)
        return context, alpha

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg


# ── Load model ────────────────────────────────────────────────────────────
MODELS_DIR = "models"
keras_path = os.path.join(MODELS_DIR, "attn_gru_final.keras")

print(f"Loading model from: {keras_path}")
model = tf.keras.models.load_model(
    keras_path,
    custom_objects={"BahdanauAttention": BahdanauAttention}
)
model.summary()


# ── Extract all weights by layer name ─────────────────────────────────────
print("\nExtracting weights...")

weights = {}

for layer in model.layers:
    w = layer.get_weights()
    if len(w) == 0:
        continue
    name = layer.name
    print(f"  {name:40s}  {[x.shape for x in w]}")

    if name == "gru_1":
        # GRU weights: [kernel (input), recurrent_kernel, bias]
        # kernel shape: (n_features, 3*units)  — for z,r,h gates
        # recurrent_kernel: (units, 3*units)
        # bias: (2, 3*units) — input bias + recurrent bias
        weights["gru1_kernel"]    = w[0]
        weights["gru1_rec_kernel"]= w[1]
        weights["gru1_bias"]      = w[2]

    elif name == "gru_2":
        weights["gru2_kernel"]    = w[0]
        weights["gru2_rec_kernel"]= w[1]
        weights["gru2_bias"]      = w[2]

    elif name == "batch_normalization":
        # gamma, beta, moving_mean, moving_var
        weights["bn1_gamma"] = w[0]
        weights["bn1_beta"]  = w[1]
        weights["bn1_mean"]  = w[2]
        weights["bn1_var"]   = w[3]

    elif name == "batch_normalization_1":
        weights["bn2_gamma"] = w[0]
        weights["bn2_beta"]  = w[1]
        weights["bn2_mean"]  = w[2]
        weights["bn2_var"]   = w[3]

    elif name == "attention":
        # W dense: kernel + bias,  V dense: kernel + bias
        weights["attn_W_kernel"] = w[0]
        weights["attn_W_bias"]   = w[1]
        weights["attn_V_kernel"] = w[2]
        weights["attn_V_bias"]   = w[3]

    elif "dense" in name and name != "log_return_pred":
        # Dense layers in the head — handle multiple
        suffix = name.replace("dense", "").replace("_", "") or "0"
        weights[f"dense{suffix}_kernel"] = w[0]
        weights[f"dense{suffix}_bias"]   = w[1]

    elif name == "log_return_pred":
        weights["out_kernel"] = w[0]
        weights["out_bias"]   = w[1]


# ── Save ──────────────────────────────────────────────────────────────────
out_path = os.path.join(MODELS_DIR, "gru_weights.npz")
np.savez(out_path, **weights)

size_kb = os.path.getsize(out_path) / 1024
print(f"\n✓ Saved {len(weights)} weight arrays → {out_path}  ({size_kb:.1f} KB)")
print("\nKeys saved:")
for k, v in weights.items():
    print(f"  {k:30s}  {v.shape}")

print("\n✓ Now upload models/gru_weights.npz to your GitHub repo.")
print("  You no longer need attn_gru_final.keras or attn_gru.tflite on Streamlit Cloud.")
