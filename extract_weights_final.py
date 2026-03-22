"""
extract_weights_final.py
=========================
Uses the EXACT paths found in your .keras file:

  layers/gru/cell/vars/0       (14, 768)   → gru1_kernel
  layers/gru/cell/vars/1       (256, 768)  → gru1_rec_kernel
  layers/gru/cell/vars/2       (2, 768)    → gru1_bias
  layers/gru_1/cell/vars/0     (256, 384)  → gru2_kernel
  layers/gru_1/cell/vars/1     (128, 384)  → gru2_rec_kernel
  layers/gru_1/cell/vars/2     (2, 384)    → gru2_bias
  (batch norm, attention, dense found by path inspection)

Usage:
    pip install h5py numpy
    python extract_weights_final.py
"""

import os, sys, zipfile, tempfile, numpy as np

try:
    import h5py
except ImportError:
    os.system(f"{sys.executable} -m pip install h5py")
    import h5py

KERAS_PATH  = os.path.join("models", "attn_gru_final.keras")
OUTPUT_PATH = os.path.join("models", "gru_weights.npz")

assert os.path.exists(KERAS_PATH), f"Not found: {KERAS_PATH}"

# ── Extract h5 from .keras zip ────────────────────────────────────────────
tmp = tempfile.mkdtemp()
with zipfile.ZipFile(KERAS_PATH, "r") as z:
    z.extractall(tmp)
h5_path = os.path.join(tmp, "model.weights.h5")

# ── Collect ALL arrays from HDF5 ─────────────────────────────────────────
all_arrays = {}
def collect(obj, path=""):
    for key in obj.keys():
        full = f"{path}/{key}" if path else key
        item = obj[key]
        if isinstance(item, h5py.Dataset):
            all_arrays[full] = np.array(item, dtype=np.float32)
        else:
            collect(item, full)

with h5py.File(h5_path, "r") as f:
    collect(f)

print("All paths found:")
for k, v in sorted(all_arrays.items()):
    print(f"  {k:80s}  {str(v.shape)}")

# ── Direct path mapping ───────────────────────────────────────────────────
def get(path):
    if path in all_arrays:
        print(f"  ✓ {path:70s} {str(all_arrays[path].shape)}")
        return all_arrays[path]
    # Try with forward/backslash variants
    alt = path.replace("/", "\\")
    if alt in all_arrays:
        print(f"  ✓ {alt:70s} {str(all_arrays[alt].shape)}")
        return all_arrays[alt]
    print(f"  ✗ NOT FOUND: {path}")
    return None

print("\nExtracting weights by exact path:")
weights = {}

# ── GRU 1 : vars/0=kernel, vars/1=rec_kernel, vars/2=bias ────────────────
v = get("layers/gru/cell/vars/0");  
if v is not None: weights["gru1_kernel"] = v

v = get("layers/gru/cell/vars/1")
if v is not None: weights["gru1_rec_kernel"] = v

v = get("layers/gru/cell/vars/2")
if v is not None: weights["gru1_bias"] = v

# ── GRU 2 ─────────────────────────────────────────────────────────────────
v = get("layers/gru_1/cell/vars/0")
if v is not None: weights["gru2_kernel"] = v

v = get("layers/gru_1/cell/vars/1")
if v is not None: weights["gru2_rec_kernel"] = v

v = get("layers/gru_1/cell/vars/2")
if v is not None: weights["gru2_bias"] = v

# ── Batch Norm 1 : vars/0=gamma,1=beta,2=moving_mean,3=moving_var ────────
for i, wkey in enumerate(["bn1_gamma","bn1_beta","bn1_mean","bn1_var"]):
    v = get(f"layers/batch_normalization/vars/{i}")
    if v is not None: weights[wkey] = v

# ── Batch Norm 2 ──────────────────────────────────────────────────────────
for i, wkey in enumerate(["bn2_gamma","bn2_beta","bn2_mean","bn2_var"]):
    v = get(f"layers/batch_normalization_1/vars/{i}")
    if v is not None: weights[wkey] = v

# ── Attention (bahdanau_attention) ────────────────────────────────────────
# Contains 2 Dense sublayers (w and v), each with kernel+bias
# Collect all attention vars and sort by size
attn_vars = {k: v for k, v in all_arrays.items()
             if "bahdanau_attention" in k.lower()}
print(f"\nAttention vars found ({len(attn_vars)}):")
for k, v in sorted(attn_vars.items()):
    print(f"  {k:70s}  {str(v.shape)}")

# Sort by size descending: W_kernel > W_bias ≈ V_kernel > V_bias
attn_sorted = sorted(attn_vars.items(), key=lambda x: x[1].size, reverse=True)

# Map: largest kernel→W_kernel, second→W_bias or V_kernel by shape
kernels = [(k,v) for k,v in attn_sorted if len(v.shape)==2]
biases  = [(k,v) for k,v in attn_sorted if len(v.shape)==1]

if len(kernels) >= 2:
    weights["attn_W_kernel"] = kernels[0][1]
    weights["attn_V_kernel"] = kernels[1][1]
    print(f"  attn_W_kernel <- {kernels[0][0]}  {kernels[0][1].shape}")
    print(f"  attn_V_kernel <- {kernels[1][0]}  {kernels[1][1].shape}")
elif len(kernels) == 1:
    weights["attn_W_kernel"] = kernels[0][1]

if len(biases) >= 2:
    weights["attn_W_bias"] = biases[0][1]
    weights["attn_V_bias"] = biases[1][1]
    print(f"  attn_W_bias   <- {biases[0][0]}  {biases[0][1].shape}")
    print(f"  attn_V_bias   <- {biases[1][0]}  {biases[1][1].shape}")
elif len(biases) == 1:
    weights["attn_W_bias"] = biases[0][1]

# ── Dense head layers ─────────────────────────────────────────────────────
# dense/vars/0=kernel, vars/1=bias
for layer_name, idx in [("dense","0"),("dense_1","1"),("dense_2","2")]:
    v = get(f"layers/{layer_name}/vars/0")
    if v is not None: weights[f"dense{idx}_kernel"] = v
    v = get(f"layers/{layer_name}/vars/1")
    if v is not None: weights[f"dense{idx}_bias"] = v

# ── Output layer (log_return_pred) ────────────────────────────────────────
# Try common names
for layer_name in ["log_return_pred", "dense_2", "dense_3"]:
    v = get(f"layers/{layer_name}/vars/0")
    if v is not None and v.ndim == 2 and v.shape[1] == 1:
        weights["out_kernel"] = v
        print(f"  out_kernel from {layer_name}")
        break

for layer_name in ["log_return_pred", "dense_2", "dense_3"]:
    v = get(f"layers/{layer_name}/vars/1")
    if v is not None and v.ndim == 1 and v.shape[0] == 1:
        weights["out_bias"] = v
        print(f"  out_bias from {layer_name}")
        break

# If still missing, scan all arrays for shape (..., 1) output layer
if "out_kernel" not in weights:
    print("\nSearching for output layer by shape (x, 1)...")
    for path, arr in sorted(all_arrays.items()):
        if arr.ndim == 2 and arr.shape[1] == 1 and arr.shape[0] <= 64:
            if "gru" not in path and "attention" not in path.lower():
                weights["out_kernel"] = arr
                print(f"  out_kernel <- {path}  {arr.shape}")
                break

if "out_bias" not in weights:
    for path, arr in sorted(all_arrays.items()):
        if arr.ndim == 1 and arr.shape[0] == 1:
            if "gru" not in path and "attention" not in path.lower():
                weights["out_bias"] = arr
                print(f"  out_bias   <- {path}  {arr.shape}")
                break

# ── Validate ──────────────────────────────────────────────────────────────
critical = ["gru1_kernel","gru1_rec_kernel","gru1_bias",
            "gru2_kernel","gru2_rec_kernel","gru2_bias",
            "out_kernel","out_bias"]
missing  = [k for k in critical if k not in weights]

print(f"\n{'='*60}")
if missing:
    print(f"⚠  Still missing: {missing}")
    print("\nAll paths in file for manual inspection:")
    for k, v in sorted(all_arrays.items()):
        print(f"  {k:80s}  {str(v.shape)}")
else:
    print(f"✓  All critical weights found!")

# ── Save ──────────────────────────────────────────────────────────────────
np.savez(OUTPUT_PATH, **weights)
size_kb = os.path.getsize(OUTPUT_PATH) / 1024
print(f"✓  Saved {len(weights)} arrays → {OUTPUT_PATH}  ({size_kb:.0f} KB)")
print("\nFinal keys:")
for k, v in sorted(weights.items()):
    print(f"  {k:30s}  {str(v.shape)}")

import shutil
shutil.rmtree(tmp, ignore_errors=True)
print("\n✓ Upload models/gru_weights.npz to your GitHub repo.")
