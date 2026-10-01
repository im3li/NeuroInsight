"""
NeuroInsight — local Flask backend.

Serves your trained 3-class model (EarlyDemented / ModerateDemented /
NonDemented) to the frontend: accepts an uploaded MRI scan, runs the
prediction, generates a Grad-CAM overlay, and builds a decision support
report — all rendered in the browser.

=====================================================================
 READ THIS BEFORE RUNNING — CONFIG YOU MUST VERIFY
=====================================================================
"""

import os
import io
import base64
import pickle

import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# ---------------------------------------------------------------------------
# CONFIG — VERIFY EVERY VALUE IN THIS BLOCK BEFORE TRUSTING ANY PREDICTION
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join("models", "resnet50_finetuned.keras")
THRESHOLD_PATH = os.path.join("models", "optimal_threshold.pkl")  # set to None if you don't have this file
IMG_SIZE = 224

# CRITICAL: This order was corrected based on evidence from your own
# DeepSeek session's classification_report output, which printed classes
# in this exact order: ModerateDemented, NonDemented, EarlyDemented.
# That is NOT alphabetical order (alphabetical would be Early/Moderate/Non),
# which means the training generator almost certainly used an explicit
# `classes=[...]` list in this sequence, not Keras's alphabetical default.
# VERIFY THIS before your presentation — see the note below.
CLASS_NAMES = ["ModerateDemented", "NonDemented", "EarlyDemented"]

# The threshold your DeepSeek session tuned (0.3232) was applied to boost
# NonDemented recall. This assumes the "argmax, but require NonDemented's
# own probability to clear this threshold before it's allowed to win"
# pattern — the most common way this kind of fix is implemented. If your
# actual threshold logic worked differently, replace apply_threshold() below
# to match exactly what you validated your 90% accuracy against — otherwise
# this app's predictions won't match the numbers in your report.
NONDEMENTED_THRESHOLD = 0.3232
NONDEMENTED_INDEX = CLASS_NAMES.index("NonDemented")

STAGE_INFO = {
    "NonDemented": {
        "display_name": "Non-Demented",
        "description": "No significant structural signs of Alzheimer's-related change were detected.",
        "recommendation": "No immediate concern indicated by this scan. Routine monitoring is still advised as part of normal care.",
        "severity": "low",
    },
    "EarlyDemented": {
        "display_name": "Early-Stage (Mild / Very Mild)",
        "description": (
            "Patterns consistent with early-stage Alzheimer's changes were detected. "
            "This model combines what were originally two separate categories (Mild and "
            "Very Mild) into one stage, since they were the hardest to reliably tell apart "
            "and merging them substantially improved the model's real-world reliability."
        ),
        "recommendation": "Consult a neurologist for a comprehensive clinical evaluation. Early-stage changes benefit most from early specialist input.",
        "severity": "moderate",
    },
    "ModerateDemented": {
        "display_name": "Moderate",
        "description": "Patterns consistent with moderate-stage Alzheimer's changes were detected.",
        "recommendation": "Consult a neurologist promptly for a comprehensive clinical evaluation and care planning.",
        "severity": "high",
    },
}

# ---------------------------------------------------------------------------

app = Flask(__name__)

print("Loading model from", MODEL_PATH, "...")
model = load_model(MODEL_PATH)
print("Model loaded. Output shape:", model.output_shape)
if model.output_shape[-1] != len(CLASS_NAMES):
    raise ValueError(
        f"Model has {model.output_shape[-1]} output classes but CLASS_NAMES "
        f"has {len(CLASS_NAMES)} entries. Fix CLASS_NAMES in app.py before continuing."
    )

optimal_threshold = None
if THRESHOLD_PATH and os.path.exists(THRESHOLD_PATH):
    with open(THRESHOLD_PATH, "rb") as f:
        optimal_threshold = pickle.load(f)
    print("Loaded threshold file. Type:", type(optimal_threshold), "Value:", optimal_threshold)
    print(">>> If this doesn't look like a single number, tell Claude the printed type/value")
    print(">>> so apply_threshold() below can be corrected to match it exactly.")
else:
    print(f"No threshold file found at {THRESHOLD_PATH} — using hardcoded NONDEMENTED_THRESHOLD = {NONDEMENTED_THRESHOLD}")


def find_last_conv_layer(keras_model):
    """Auto-detects the last Conv2D layer for Grad-CAM, works across architectures."""
    for layer in reversed(keras_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model — cannot generate Grad-CAM.")


LAST_CONV_LAYER = find_last_conv_layer(model)
print("Grad-CAM will use conv layer:", LAST_CONV_LAYER)


def apply_threshold(probs):
    """
    Applies the tuned NonDemented threshold. Handles a few common formats
    the .pkl file might actually be in, since the exact original tuning
    code wasn't available when this was written:
      - a plain float/np number -> used directly as the NonDemented bar
      - a dict keyed by class name or index -> looks up NonDemented's value
      - a list/array of per-class thresholds -> takes the NonDemented entry
    If none of these match, falls back to the hardcoded NONDEMENTED_THRESHOLD.
    """
    threshold = NONDEMENTED_THRESHOLD
    if optimal_threshold is not None:
        t = optimal_threshold
        if isinstance(t, (int, float, np.floating)):
            threshold = float(t)
        elif isinstance(t, dict):
            if "NonDemented" in t:
                threshold = float(t["NonDemented"])
            elif NONDEMENTED_INDEX in t:
                threshold = float(t[NONDEMENTED_INDEX])
            else:
                print(f"WARNING: threshold dict keys {list(t.keys())} don't contain 'NonDemented' "
                      f"or index {NONDEMENTED_INDEX} — using fallback {NONDEMENTED_THRESHOLD}")
        elif isinstance(t, (list, tuple, np.ndarray)) and len(t) == len(CLASS_NAMES):
            threshold = float(t[NONDEMENTED_INDEX])
        else:
            print(f"WARNING: unrecognized threshold format ({type(t)}) — using fallback {NONDEMENTED_THRESHOLD}")


    if probs[NONDEMENTED_INDEX] >= threshold:
        return NONDEMENTED_INDEX
    other_indices = [i for i in range(len(probs)) if i != NONDEMENTED_INDEX]
    best_other = max(other_indices, key=lambda i: probs[i])
    return best_other


        # Inside apply_threshold, add a print:
    print(f"apply_threshold: using threshold = {threshold}")
    print(f"NonDemented prob: {probs[NONDEMENTED_INDEX]}")

def preprocess_image(pil_img):
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = tf.keras.utils.img_to_array(img)
    arr = preprocess_input(np.expand_dims(arr, axis=0))
    return arr, img


def generate_gradcam(img_array, pred_index):
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(LAST_CONV_LAYER).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap.numpy()


def overlay_heatmap(pil_img, heatmap):
    img_np = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img_bgr, 0.6, heatmap_colored, 0.4, 0)
    superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
    return Image.fromarray(superimposed_rgb)


def pil_to_base64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    print("CLASS_NAMES currently:", CLASS_NAMES)
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    try:
        pil_img = Image.open(file.stream)
    except Exception:
        return jsonify({"error": "Could not read the uploaded file as an image"}), 400

    img_array, resized_img = preprocess_image(pil_img)

    raw_probs = model.predict(img_array, verbose=0)[0]
    print("\n" + "="*50)
    print(f"RAW PROBS: {dict(zip(CLASS_NAMES, raw_probs))}")
    print(f"NONDEMENTED_INDEX: {NONDEMENTED_INDEX}, THRESHOLD: {NONDEMENTED_THRESHOLD}")
    pred_index = apply_threshold(raw_probs)
    print(f"PRED_INDEX: {pred_index} -> CLASS: {CLASS_NAMES[pred_index]}")
    print("="*50)



    raw_probs = model.predict(img_array, verbose=0)[0]
    pred_index = apply_threshold(raw_probs)
    pred_class = CLASS_NAMES[pred_index]
    confidence = float(raw_probs[pred_index])

    heatmap = generate_gradcam(img_array, pred_index)
    overlay_img = overlay_heatmap(resized_img, heatmap)

    stage = STAGE_INFO[pred_class]

    response = {
        "prediction": pred_class,
        "display_name": stage["display_name"],
        "confidence": round(confidence * 100, 1),
        "description": stage["description"],
        "recommendation": stage["recommendation"],
        "severity": stage["severity"],
        "nondemented_threshold": float(optimal_threshold) if isinstance(optimal_threshold, (int, float, np.floating)) else NONDEMENTED_THRESHOLD,
        "class_probabilities": [
            {
                "class": CLASS_NAMES[i],
                "display_name": STAGE_INFO[CLASS_NAMES[i]]["display_name"],
                "probability": round(float(raw_probs[i]) * 100, 1),
            }
            for i in range(len(CLASS_NAMES))
        ],
        "original_image": pil_to_base64(resized_img),
        "gradcam_image": pil_to_base64(overlay_img),
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
