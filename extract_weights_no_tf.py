"""
extract_weights_no_tf.py
========================
Extracts GRU weights from attn_gru_final.keras WITHOUT needing TensorFlow.

A .keras file is just a ZIP archive containing:
  - config.json        (architecture)
  - model.weights.h5   (all weight arrays in HDF5 format)

We read the HDF5 file directly using h5py.

Usage:
    pip install h5py numpy
    python extract_weights_no_tf.py

Output:
    models/gru_weights.npz
"""

import os
import sys
import zipfile
import tempfile
import json
import numpy as np

# ── Check h5py available ───────────────────────────────────────────────────
try:
    import h5py
except ImportError:
    print("Installing h5py...")
    os.system(f"{sys.executable} -m pip install h5py")
    import h5py

# ── Paths ──────────────────────────────────────────────────────────────────
KERAS_PATH  = os.path.join("models", "attn_gru_final.keras")
OUTPUT_PATH = os.path.join("models", "gru_weights.npz")

if not os.path.exists(KERAS_PATH):
    print(f"ERROR: {KERAS_PATH} not found.")
    print("Make sure you run this script from the reliance_gru_app folder.")
    sys.exit(1)

print(f"Reading: {KERAS_PATH}")
print(f"File size: {os.path.getsize(KERAS_PATH)/1024:.1f} KB")

# ── Extract the .keras ZIP to a temp folder ────────────────────────────────
tmp_dir = tempfile.mkdtemp()
with zipfile.ZipFile(KERAS_PATH, "r") as z:
    names = z.namelist()
    print(f"\nContents of .keras file:")
    for n in names:
        print(f"  {n}")
    z.extractall(tmp_dir)

# ── Find the weights file ──────────────────────────────────────────────────
# Keras 3 format: model.weights.h5
# Keras 2 format: saved_model.pb or variables/
h5_candidates = [
    os.path.join(tmp_dir, "model.weights.h5"),
    os.path.join(tmp_dir, "weights.h5"),
]
# Also search recursively
for root, dirs, files in os.walk(tmp_dir):
    for f in files:
        if f.endswith(".h5") or f.endswith(".hdf5"):
            h5_candidates.append(os.path.join(root, f))

h5_path = None
for c in h5_candidates:
    if os.path.exists(c):
        h5_path = c
        break

if h5_path is None:
    print("\nERROR: Could not find .h5 weights file inside .keras archive.")
    print("Files found:")
    for root, dirs, files in os.walk(tmp_dir):
        for f in files:
            print(f"  {os.path.join(root, f)}")
    sys.exit(1)

print(f"\nFound weights file: {h5_path}")
print(f"Size: {os.path.getsize(h5_path)/1024:.1f} KB")

# ── Inspect the HDF5 structure ─────────────────────────────────────────────
def print_hdf5_structure(h5file, prefix=""):
    for key in h5file.keys():
        item = h5file[key]
        if isinstance(item, h5py.Dataset):
            print(f"  {prefix}{key}  {item.shape}  {item.dtype}")
        else:
            print(f"  {prefix}{key}/")
            print_hdf5_structure(item, prefix + "  ")

print("\nHDF5 structure:")
with h5py.File(h5_path, "r") as f:
    print_hdf5_structure(f)


# ── Extract weights by layer path ─────────────────────────────────────────
def get_all_datasets(h5file):
    """Recursively collect all dataset paths and arrays."""
    datasets = {}
    def _recurse(obj, path=""):
        for key in obj.keys():
            full = f"{path}/{key}" if path else key
            item = obj[key]
            if isinstance(item, h5py.Dataset):
                datasets[full] = np.array(item)
            else:
                _recurse(item, full)
    _recurse(h5file)
    return datasets

print("\nExtracting all weight arrays...")
with h5py.File(h5_path, "r") as f:
    all_w = get_all_datasets(f)

print(f"Found {len(all_w)} arrays total:")
for k, v in sorted(all_w.items()):
    print(f"  {k:70s}  {str(v.shape):20s}  {v.dtype}")


# ── Map arrays to named keys ────────────────────────────────────────────────
# The key names inside the HDF5 depend on which Keras version saved the model.
# We match by searching for keywords in the path.

weights = {}

def find(keyword_list, arrays, exclude=None):
    """Find arrays whose path contains ALL keywords, optionally excluding some."""
    exclude = exclude or []
    matches = {}
    for path, arr in arrays.items():
        p = path.lower()
        if all(k.lower() in p for k in keyword_list):
            if not any(e.lower() in p for e in exclude):
                matches[path] = arr
    return matches


def pick(keyword_list, arrays, exclude=None, prefer_longer=False):
    """Pick a single array matching keywords. Returns (path, array) or None."""
    m = find(keyword_list, arrays, exclude)
    if not m:
        return None, None
    if prefer_longer:
        path = max(m, key=lambda p: m[p].size)
    else:
        path = sorted(m.keys())[0]
    return path, m[path]


# ── GRU 1 ──────────────────────────────────────────────────────────────────
# Keras stores GRU weights as: kernel, recurrent_kernel, bias
p, v = pick(["gru_1", "kernel"],            all_w, exclude=["recurrent","bias"])
if v is not None: weights["gru1_kernel"] = v;     print(f"gru1_kernel      <- {p}  {v.shape}")

p, v = pick(["gru_1", "recurrent_kernel"],  all_w, exclude=["bias"])
if v is not None: weights["gru1_rec_kernel"] = v; print(f"gru1_rec_kernel  <- {p}  {v.shape}")

p, v = pick(["gru_1", "bias"],              all_w)
if v is not None: weights["gru1_bias"] = v;       print(f"gru1_bias        <- {p}  {v.shape}")

# ── GRU 2 ──────────────────────────────────────────────────────────────────
p, v = pick(["gru_2", "kernel"],            all_w, exclude=["recurrent","bias"])
if v is not None: weights["gru2_kernel"] = v;     print(f"gru2_kernel      <- {p}  {v.shape}")

p, v = pick(["gru_2", "recurrent_kernel"],  all_w, exclude=["bias"])
if v is not None: weights["gru2_rec_kernel"] = v; print(f"gru2_rec_kernel  <- {p}  {v.shape}")

p, v = pick(["gru_2", "bias"],              all_w)
if v is not None: weights["gru2_bias"] = v;       print(f"gru2_bias        <- {p}  {v.shape}")

# ── Batch Norm 1 ────────────────────────────────────────────────────────────
bn1_all = find(["batch_normalization", "gamma"], all_w, exclude=["_1"])
if not bn1_all:
    bn1_all = find(["batch_normalization"], all_w, exclude=["_1","_2"])

p, v = pick(["batch_normalization", "gamma"],         all_w, exclude=["_1"])
if v is not None: weights["bn1_gamma"] = v; print(f"bn1_gamma        <- {p}")
p, v = pick(["batch_normalization", "beta"],          all_w, exclude=["_1"])
if v is not None: weights["bn1_beta"]  = v; print(f"bn1_beta         <- {p}")
p, v = pick(["batch_normalization", "moving_mean"],   all_w, exclude=["_1"])
if v is not None: weights["bn1_mean"]  = v; print(f"bn1_mean         <- {p}")
p, v = pick(["batch_normalization", "moving_var"],    all_w, exclude=["_1"])
if v is not None: weights["bn1_var"]   = v; print(f"bn1_var          <- {p}")

# ── Batch Norm 2 ────────────────────────────────────────────────────────────
p, v = pick(["batch_normalization_1", "gamma"],       all_w)
if v is not None: weights["bn2_gamma"] = v; print(f"bn2_gamma        <- {p}")
p, v = pick(["batch_normalization_1", "beta"],        all_w)
if v is not None: weights["bn2_beta"]  = v; print(f"bn2_beta         <- {p}")
p, v = pick(["batch_normalization_1", "moving_mean"], all_w)
if v is not None: weights["bn2_mean"]  = v; print(f"bn2_mean         <- {p}")
p, v = pick(["batch_normalization_1", "moving_var"],  all_w)
if v is not None: weights["bn2_var"]   = v; print(f"bn2_var          <- {p}")

# ── Attention (W dense + V dense) ───────────────────────────────────────────
attn_kernels = find(["attention"], all_w, exclude=["bias"])
attn_biases  = find(["attention"], all_w, exclude=["kernel"])

attn_k_sorted = sorted(attn_kernels.items(), key=lambda x: x[1].size, reverse=True)
attn_b_sorted = sorted(attn_biases.items(),  key=lambda x: x[1].size, reverse=True)

if len(attn_k_sorted) >= 2:
    weights["attn_W_kernel"] = attn_k_sorted[0][1]
    weights["attn_V_kernel"] = attn_k_sorted[1][1]
    print(f"attn_W_kernel    <- {attn_k_sorted[0][0]}  {attn_k_sorted[0][1].shape}")
    print(f"attn_V_kernel    <- {attn_k_sorted[1][0]}  {attn_k_sorted[1][1].shape}")
elif len(attn_k_sorted) == 1:
    weights["attn_W_kernel"] = attn_k_sorted[0][1]

if len(attn_b_sorted) >= 2:
    weights["attn_W_bias"] = attn_b_sorted[0][1]
    weights["attn_V_bias"] = attn_b_sorted[1][1]
    print(f"attn_W_bias      <- {attn_b_sorted[0][0]}")
    print(f"attn_V_bias      <- {attn_b_sorted[1][0]}")
elif len(attn_b_sorted) == 1:
    weights["attn_W_bias"] = attn_b_sorted[0][1]

# ── Dense head layers ────────────────────────────────────────────────────────
# Find all dense layers that are NOT attention, NOT output
dense_kernels = {
    p: v for p, v in all_w.items()
    if "dense" in p.lower()
    and "kernel" in p.lower()
    and "attention" not in p.lower()
    and "log_return" not in p.lower()
}
dense_biases = {
    p: v for p, v in all_w.items()
    if "dense" in p.lower()
    and "bias" in p.lower()
    and "attention" not in p.lower()
    and "log_return" not in p.lower()
}

dense_k_sorted = sorted(dense_kernels.items())
dense_b_sorted = sorted(dense_biases.items())

for i, ((pk, vk), (pb, vb)) in enumerate(zip(dense_k_sorted, dense_b_sorted)):
    weights[f"dense{i}_kernel"] = vk
    weights[f"dense{i}_bias"]   = vb
    print(f"dense{i}_kernel      <- {pk}  {vk.shape}")
    print(f"dense{i}_bias        <- {pb}  {vb.shape}")

# ── Output layer ─────────────────────────────────────────────────────────────
p, v = pick(["log_return_pred", "kernel"], all_w)
if v is None:
    p, v = pick(["log_return", "kernel"], all_w)
if v is not None: weights["out_kernel"] = v; print(f"out_kernel       <- {p}  {v.shape}")

p, v = pick(["log_return_pred", "bias"], all_w)
if v is None:
    p, v = pick(["log_return", "bias"], all_w)
if v is not None: weights["out_bias"] = v;   print(f"out_bias         <- {p}  {v.shape}")


# ── Validate we got the critical weights ──────────────────────────────────
critical = ["gru1_kernel", "gru1_rec_kernel", "gru1_bias",
            "gru2_kernel", "gru2_rec_kernel", "gru2_bias",
            "out_kernel",  "out_bias"]
missing = [k for k in critical if k not in weights]

if missing:
    print(f"\n⚠  WARNING: Missing critical weights: {missing}")
    print("   The extraction may have failed due to an unexpected HDF5 structure.")
    print("   Please share the HDF5 structure printed above for debugging.")
else:
    print(f"\n✓ All critical weights extracted ({len(weights)} arrays total)")


# ── Save ──────────────────────────────────────────────────────────────────
np.savez(OUTPUT_PATH, **weights)
size_kb = os.path.getsize(OUTPUT_PATH) / 1024
print(f"✓ Saved → {OUTPUT_PATH}  ({size_kb:.1f} KB)")
print("\nKeys in gru_weights.npz:")
for k, v in weights.items():
    print(f"  {k:30s}  {str(v.shape):20s}  {v.dtype}")

print("\n✓ Upload models/gru_weights.npz to your GitHub repo.")
print("  TensorFlow is no longer needed.")

# Cleanup temp
import shutil
shutil.rmtree(tmp_dir, ignore_errors=True)
