# app.py
# Brain MRI — Alzheimer's Stage Classifier (EffNetV2-B0, 300x300)
from pathlib import Path
import json
import numpy as np
from PIL import Image

import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as v2_preproc

# ---------- constants ----------
APP_DIR = Path(__file__).parent.resolve()
KERAS_PATH = APP_DIR / "brain_effv2b0_infer.keras"
LABELS_PATH = APP_DIR / "labels.json"
SAVEDMODEL_DIR = APP_DIR / "brain_savedmodel"
IMG_SZ = 300

st.set_page_config(page_title="Brain MRI Classifier", layout="wide")
st.title("🧠 Brain MRI — Alzheimer's Stage Classifier")

# ---------- utilities ----------
def load_labels(path: Path):
    lab = json.loads(path.read_text(encoding="utf-8"))
    return [lab[str(i)] if str(i) in lab else lab[i] for i in range(len(lab))]

def preprocess_pil(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMG_SZ, IMG_SZ))
    x = np.array(img, dtype=np.float32)[None, ...]
    return v2_preproc(x)

def predict_one(model, img: Image.Image) -> np.ndarray:
    x = preprocess_pil(img)
    return model.predict(x, verbose=0)  # (1, C)

def get_last_conv_name(keras_model):
    last = None
    for lyr in getattr(keras_model, "layers", []):
        if isinstance(lyr, tf.keras.layers.Conv2D):
            last = lyr.name
    return last

def grad_cam(keras_model, img: Image.Image, alpha=0.40):
    try:
        if not hasattr(keras_model, "layers") or not keras_model.layers:
            return None
        layer_name = get_last_conv_name(keras_model)
        if layer_name is None:
            return None

        x = preprocess_pil(img)
        x_tensor = tf.constant(x, dtype=tf.float32)

        grad_model = tf.keras.Model(
            inputs=keras_model.inputs,
            outputs=[keras_model.get_layer(layer_name).output, keras_model.output]
        )

        # --- Step 1: Run once OUTSIDE tape to safely get class index ---
        conv_out_val, preds_val = grad_model(x_tensor, training=False)
        preds_np = np.array(preds_val)          # safe numpy conversion
        class_idx = int(np.argmax(preds_np[0])) # plain Python int
        num_classes = preds_np.shape[-1]         # plain Python int

        # --- Step 2: Build one-hot as a numpy array (no TF shape issues) ---
        one_hot_np = np.zeros((1, num_classes), dtype=np.float32)
        one_hot_np[0, class_idx] = 1.0
        one_hot = tf.constant(one_hot_np)

        # --- Step 3: Run INSIDE tape, watch conv output directly ---
        with tf.GradientTape() as tape:
            conv_out_tape, preds_tape = grad_model(x_tensor, training=False)
            tape.watch(conv_out_tape)
            loss = tf.reduce_sum(preds_tape * one_hot)

        grads = tape.gradient(loss, conv_out_tape)
        if grads is None:
            st.warning("Grad-CAM: gradients were None.")
            return None

        # --- Step 4: Convert everything to NumPy for safety ---
        grads_np = np.array(grads[0])       # (H, W, C)
        conv_np  = np.array(conv_out_tape[0])  # (H, W, C)

        weights = np.mean(grads_np, axis=(0, 1))          # (C,)
        cam = np.sum(weights * conv_np, axis=-1)           # (H, W)
        cam = np.maximum(cam, 0)
        cam_max = np.max(cam)
        if cam_max == 0:
            return None
        cam = cam / (cam_max + 1e-8)

        # --- Step 5: Resize using PIL (no TF needed) ---
        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (IMG_SZ, IMG_SZ), Image.BILINEAR
        )
        cam_resized = np.array(cam_img) / 255.0

        base    = np.array(img.convert("RGB").resize((IMG_SZ, IMG_SZ)), np.float32) / 255.0
        heat    = plt.cm.jet(cam_resized)[..., :3]
        overlay = np.clip((1 - alpha) * base + alpha * heat, 0, 1)
        return (overlay * 255).astype(np.uint8)

    except Exception as e:
        st.warning(f"Grad-CAM failed: {str(e)}")
        return None

# ---------- load model & labels ----------
try:
    class_names = load_labels(LABELS_PATH)
except Exception as e:
    st.error(f"Could not read labels.json: {e}")
    st.stop()

try:
    if KERAS_PATH.exists():
        model = load_model(str(KERAS_PATH),
                           custom_objects={"custom>effv2_preproc": v2_preproc})
        is_keras = True
        st.success("Loaded Keras model.")
    elif (SAVEDMODEL_DIR / "saved_model.pb").exists():
        infer = tf.saved_model.load(str(SAVEDMODEL_DIR))
        serving = infer.signatures["serving_default"]
        class Wrap:
            def predict(self, x, verbose=0):
                out = serving(tf.constant(x))
                return next(iter(out.values())).numpy()
            @property
            def layers(self):
                return []
        model = Wrap()
        is_keras = False
        st.success("Loaded SavedModel.")
    else:
        raise FileNotFoundError("No model found. Place 'brain_effv2b0_infer.keras' "
                                "or 'brain_savedmodel/' next to this file.")
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

st.caption("Classes: " + " | ".join(class_names))
st.markdown("---")

# ---------- UI: upload & predict ----------
show_cam = st.checkbox("Show Grad-CAM (only for .keras models)", value=True)
file = st.file_uploader("Upload an MRI image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file)
    st.image(img, caption="Input", use_container_width=True)

    probs = predict_one(model, img)[0]
    top = int(np.argmax(probs))
    st.subheader(f"Predicted Stage: **{class_names[top]}** (confidence {probs[top]:.3f})")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(len(class_names)), probs)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    st.pyplot(fig, use_container_width=True)

    if show_cam and is_keras:
        with st.spinner("Generating Grad-CAM..."):
            overlay = grad_cam(model, img)
        if overlay is not None:
            st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)
        else:
            st.info("Grad-CAM could not be generated for this image.")
