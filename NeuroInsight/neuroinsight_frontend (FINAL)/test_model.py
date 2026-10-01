# test_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import pickle
import os

# Load model and threshold
model = tf.keras.models.load_model("models/resnet50_finetuned.keras")
with open("models/optimal_threshold.pkl", "rb") as f:
    threshold_data = pickle.load(f)
# extract threshold value (handles different types)
if isinstance(threshold_data, dict):
    threshold = threshold_data.get('threshold', 0.5)
elif isinstance(threshold_data, (list, tuple)):
    threshold = threshold_data[0] if len(threshold_data) > 0 else 0.5
else:
    threshold = float(threshold_data)

# Class order – must match what the model was trained on
CLASS_NAMES = ["ModerateDemented", "NonDemented", "EarlyDemented"]
NONDEMENTED_INDEX = 1

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    probs = model.predict(x, verbose=0)[0]
    
    # Apply threshold logic (same as in app.py)
    if probs[NONDEMENTED_INDEX] >= threshold:
        pred_idx = NONDEMENTED_INDEX
    else:
        # Choose max among the other two
        other_probs = np.delete(probs, NONDEMENTED_INDEX)
        pred_idx = np.argmax(other_probs)
        if pred_idx >= NONDEMENTED_INDEX:
            pred_idx += 1
    confidence = probs[pred_idx]
    pred_class = CLASS_NAMES[pred_idx]
    return pred_class, confidence, probs

if __name__ == "__main__":
    # Replace with actual image paths (you can put them in a folder and list)
    from pathlib import Path
    base = Path("C:/Users/3lisu/Desktop/NeuroInsight/Datasets/archive3/train/ModerateDemented")
    test_images = list(base.glob("*.jpg"))[:4]   # takes first 4 jpgs from that folder
    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        pred_class, conf, probs = predict_image(img_path)
        print(f"{img_path}: {pred_class} ({conf:.2%})")
        print(f"  Full probs: {dict(zip(CLASS_NAMES, probs))}\n")