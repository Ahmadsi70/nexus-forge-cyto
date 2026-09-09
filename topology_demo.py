#!/usr/bin/env python3
"""
TopoNet — Interactive Topology Previewer (Streamlit).

A free, open-source complement to FEA topology optimization solvers.
Adjust physical parameters (volume fraction, load position, load
direction) and get an instant binary topology prediction in ~3 seconds.

Run:
    pip install streamlit pillow onnxruntime numpy
    streamlit run topology_demo.py
"""
import time
import numpy as np
import streamlit as st
import onnxruntime as ort
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "models" / "hovernet_topology_seg.onnx"

st.set_page_config(page_title="TopoNet — Topology Previewer", page_icon="🧊", layout="wide")

st.title("🧊 TopoNet — Instant Topology Previewer")
st.caption(
    "**Complementary tool** for structural optimization: predict a binary material/void "
    "topology from physical field inputs in seconds — a warm start for FEA solvers "
    "(Abaqus, Ansys, OptiStruct). Trained on MIT TopoDiff (Dice 0.849)."
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model not found: {MODEL_PATH}")
        st.stop()
    sess = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    return sess


def synthesize_fields(vf, load_x, load_y, load_mag, noise, complexity):
    """
    Synthesize a 3-channel physical field (256x256) approximating a topology
    optimization load case:
      ch0: strain energy density (high near load boundary, decays with distance)
      ch1: volume fraction (constant)
      ch2: load distribution (gaussian around load point)
    """
    yy, xx = np.mgrid[0:256, 0:256]

    # ch0 — strain energy: decays from the load-side boundary
    dist_from_load_side = np.sqrt((xx - load_x) ** 2 + (yy - load_y) ** 2)
    # Primary field: gaussian around load point + boundary gradient
    strain = np.exp(-(dist_from_load_side ** 2) / (2 * (60 + complexity * 8) ** 2))
    # Add structural wave pattern (like a truss-like energy distribution)
    wave = np.abs(np.sin(xx / (10 + complexity) + yy / (12 + complexity))) * 0.35
    strain = strain * 0.65 + wave * (load_mag / 10.0)
    strain /= strain.max() + 1e-9

    # ch1 — volume fraction (constant per sample)
    vf_ch = np.full((256, 256), vf, dtype=np.float32)

    # ch2 — load magnitude map
    load_map = np.exp(-(dist_from_load_side ** 2) / (2 * 40 ** 2)) * (load_mag / 10.0)

    ch0 = (strain * 255).astype(np.uint8)
    ch1 = (np.clip(vf_ch, 0, 1) * 255).astype(np.uint8)
    ch2 = (np.clip(load_map, 0, 1) * 255).astype(np.uint8)

    # Noise
    if noise > 0:
        rng = np.random.default_rng(0)
        ch0 = np.clip(ch0.astype(np.float32) + rng.normal(0, noise * 30, ch0.shape), 0, 255).astype(np.uint8)

    img = np.stack([ch0, ch1, ch2], axis=-1).astype(np.float32)  # (256,256,3)
    return img


def predict(sess, img):
    x = np.ascontiguousarray(img.transpose(2, 0, 1)[None])  # (1,3,256,256)
    t0 = time.time()
    outs = sess.run(None, {sess.get_inputs()[0].name: x})
    dt = time.time() - t0
    np_logits = outs[0]  # (1,2,164,164)
    # softmax
    e = np.exp(np_logits - np_logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    return probs[0, 1], dt  # foreground prob (164,164)


def main():
    sess = load_model()

    # ── Sidebar controls ─────────────────────────────────────────────
    st.sidebar.header("⚙️ Physical Parameters")
    vf = st.sidebar.slider("Volume Fraction (material budget)", 0.10, 0.60, 0.30, 0.01)
    load_mag = st.sidebar.slider("Load Magnitude", 1.0, 20.0, 10.0, 0.5)
    load_x = st.sidebar.slider("Load Position X", 20, 236, 200, 1)
    load_y = st.sidebar.slider("Load Position Y", 20, 236, 128, 1)
    noise = st.sidebar.slider("Field Noise", 0.0, 1.0, 0.15, 0.05)
    complexity = st.sidebar.slider("Structural Complexity", 1, 10, 5, 1)
    threshold = st.sidebar.slider("Binarization Threshold", 0.30, 0.70, 0.50, 0.01)
    run = st.sidebar.button("🎯 Predict Topology", type="primary")

    # ── Main panel ───────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    if "last_img" not in st.session_state:
        st.session_state.last_img = None
        st.session_state.last_probs = None

    img = synthesize_fields(vf, load_x, load_y, load_mag, noise, complexity)

    with c1:
        st.subheader("📡 Input Physical Fields")
        show = np.stack([
            img[..., 0] * 0.5, img[..., 1] * 0.5, img[..., 2] * 0.7
        ], axis=-1).astype(np.uint8)
        # brighten to be more visible
        st.image(show, caption="ch0=strain energy, ch1=volume fraction, ch2=load", use_container_width=True)

    with c2:
        st.subheader("🧊 Predicted Topology")
        if run or st.session_state.last_probs is None:
            with st.spinner("Running inference (~3s)..."):
                probs, dt = predict(sess, img)
            st.session_state.last_img = img
            st.session_state.last_probs = probs
            st.session_state.last_dt = dt
            st.session_state.last_thresh = threshold
            st.session_state.last_vf = vf
            st.session_state.last_load = load_mag

        probs = st.session_state.last_probs
        mask = (probs > threshold).astype(np.uint8) * 255
        st.image(mask, caption=f"Binary mask @ θ={threshold}", use_container_width=True, clamp=True)

        # Metrics
        m1, m2, m3 = st.columns(3)
        fg = (probs > threshold).mean()
        m1.metric("Foreground", f"{fg*100:.1f}%")
        m2.metric("Inference", f"{st.session_state.get('last_dt', 0)*1000:.0f} ms")
        m3.metric("Binarized", "✓" if 0.05 < fg < 0.95 else "⚠️")

        # Volume fraction comparison
        st.caption(
            f"Requested vf={st.session_state.get('last_vf', 0.3):.2f} | "
            f"Predicted fg={fg:.2f}. In a real optimization loop this prediction "
            f"would seed a solver, cutting convergence time."
        )

    st.divider()
    st.caption(
        "⚠️ **Experimental tool.** Predictions are a fast initial guess, not a validated "
        "FEA result. Always verify critical designs with a full structural analysis. "
        "Model trained on MIT TopoDiff samples (physical fields → topology)."
    )


if __name__ == "__main__":
    main()