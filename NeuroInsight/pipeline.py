import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shutil
import random
import json
import warnings
warnings.filterwarnings('ignore')

# ------------------- CONFIGURATION -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_SRC = os.path.join(BASE_DIR, "datasets")
CLEANED_DST = os.path.join(BASE_DIR, "cleaned_dataset")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

CLASSES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_PHASE1 = 25   # frozen backbone, training the head
EPOCHS_PHASE2 = 15   # fine-tuning, top layers unfrozen at low LR

# Perceptual-hash distance below which two images count as near-duplicates.
# Lowered from 5 to 3: at 5, clustering ACROSS all classes at once "chained"
# genuinely distinct images together (A~B~C even though A and C aren't
# actually similar), producing a few giant, mostly-fictional clusters.
# Clustering is now also done PER LABEL (see _cluster_within_label), which
# independently reduces false merges since we never compare across classes.
HAMMING_THRESHOLD = 3

STAGE_MAP = {
    'NonDemented': "No significant signs detected",
    'VeryMildDemented': "Early Alzheimer's Progression (Stage 1 of 4)",
    'MildDemented': "Mild Alzheimer's Progression (Stage 2 of 4)",
    'ModerateDemented': "Moderate Alzheimer's Progression (Stage 3 of 4)",
}

os.makedirs(CLEANED_DST, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------- STEP 1: DATA ANALYSIS ----------------
def run_analysis():
    print("\n===== STEP 1: Analyzing Datasets =====")
    dataset_names = ["AugmentedAlzheimerDataset", "OriginalDataset", "Archive2", "Archive3"]
    data = []

    for name in dataset_names:
        for cls in CLASSES:
            count = 0
            if name == "Archive3":
                p1 = os.path.join(DATASET_SRC, name, "train", cls)
                p2 = os.path.join(DATASET_SRC, name, "val", cls)
                if os.path.exists(p1): count += len(os.listdir(p1))
                if os.path.exists(p2): count += len(os.listdir(p2))
            else:
                p = os.path.join(DATASET_SRC, name, cls)
                if os.path.exists(p): count = len(os.listdir(p))
            data.append([name, cls, count])

    df = pd.DataFrame(data, columns=['Dataset', 'Class', 'Count'])
    pivot = df.pivot(index='Dataset', columns='Class', values='Count')
    print(pivot)

    pivot.plot(kind='bar', figsize=(10, 6))
    plt.title("Alzheimer Dataset Distribution")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("distribution_plot.png")
    plt.show()
    print("Plot saved as 'distribution_plot.png'")


# ---------------- STEP 2: CLEANING ----------------
def run_cleaning():
    print("\n===== STEP 2: Cleaning Datasets (Removing bad files & duplicates) =====")
    from PIL import Image
    import imagehash
    from tqdm import tqdm

    for root, dirs, files in os.walk(DATASET_SRC):
        for f in tqdm(files, desc="Checking Corrupt"):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                fp = os.path.join(root, f)
                try:
                    img = Image.open(fp)
                    img.verify()
                except Exception:
                    os.remove(fp)
                    print(f"Removed corrupt: {fp}")

    hashes = {}
    for root, dirs, files in os.walk(DATASET_SRC):
        for f in tqdm(files, desc="Checking Duplicates"):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                fp = os.path.join(root, f)
                try:
                    img = Image.open(fp)
                    h = imagehash.phash(img)
                    if h in hashes:
                        os.remove(fp)
                    else:
                        hashes[h] = fp
                except:
                    continue
    print("Cleaning complete!")


# ---------------- STEP 3: MERGE / FUSION ----------------
def run_merge():
    print("\n===== STEP 3: Merging datasets into 'cleaned_dataset' =====")
    for c in CLASSES:
        os.makedirs(os.path.join(CLEANED_DST, c), exist_ok=True)

    dataset_names = ["AugmentedAlzheimerDataset", "OriginalDataset", "Archive2", "Archive3"]

    for name in dataset_names:
        for cls in CLASSES:
            if name == "Archive3":
                paths = [os.path.join(DATASET_SRC, name, "train", cls),
                         os.path.join(DATASET_SRC, name, "val", cls)]
            else:
                paths = [os.path.join(DATASET_SRC, name, cls)]

            for src_path in paths:
                if os.path.exists(src_path):
                    for img in os.listdir(src_path):
                        if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                            src_file = os.path.join(src_path, img)
                            dst_file = os.path.join(CLEANED_DST, cls, img)
                            if not os.path.exists(dst_file):
                                shutil.copy2(src_file, dst_file)

    print("Merge complete! Final counts:")
    for c in CLASSES:
        print(f"{c}: {len(os.listdir(os.path.join(CLEANED_DST, c)))} images")


# ---------------- Union-Find helper for near-duplicate clustering ----------------
class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[ry] = rx


def _hash_all_images(df):
    """Perceptual-hash every image in df['filename']. Returns df with a 'hash' column, bad rows dropped."""
    from PIL import Image
    import imagehash
    from tqdm import tqdm

    hashes = []
    for path in tqdm(df['filename'], desc="Hashing images"):
        try:
            hashes.append(imagehash.phash(Image.open(path)))
        except Exception:
            hashes.append(None)
    df = df.copy()
    df['hash'] = hashes
    df = df[df['hash'].notnull()].reset_index(drop=True)
    return df


def _cluster_within_label(df_label, threshold=HAMMING_THRESHOLD, block_size=2000):
    """
    Vectorized (NumPy matmul) near-duplicate clustering, run separately for
    EACH class. This replaces a pure-Python O(n^2) double loop that took
    ~3 hours across all 98,971 images combined. By clustering one class at
    a time and using blocked matrix multiplication for the Hamming distance
    (dist = popcount_i + popcount_j - 2*shared_bits), this finishes in a
    small fraction of the time -- and clustering within a single class only
    (rather than across all 4 at once) avoids the "chaining" bug where
    unrelated images got merged transitively into a few giant clusters.
    """
    from tqdm import tqdm

    n = len(df_label)
    bit_arrays = np.stack([h.hash.flatten() for h in df_label['hash']]).astype(np.float32)
    popcounts = bit_arrays.sum(axis=1)

    uf = _UnionFind(n)

    for start in tqdm(range(0, n, block_size), desc="Clustering (vectorized)"):
        end = min(start + block_size, n)
        block = bit_arrays[start:end]           # (b, bits)
        rest = bit_arrays[start:]               # (r, bits) -- only compare forward, mirrors i<j
        dot = block @ rest.T                    # (b, r) shared-1-bit counts
        dist = popcounts[start:end, None] + popcounts[None, start:] - 2 * dot
        close_pairs = np.argwhere(dist <= threshold)
        for bi, rj in close_pairs:
            i = start + int(bi)
            j = start + int(rj)
            if j > i:
                uf.union(i, j)

    df_label = df_label.copy()
    df_label['cluster'] = [uf.find(i) for i in range(n)]
    return df_label


# ---------------- STEP 4: LEAKAGE-SAFE, STRATIFIED, BALANCED SPLIT ----------------
def run_split():
    """
    Per-label clustering + per-label splitting. Fixes two problems found in
    the previous run:
      1. Cross-class clustering caused "chaining" -- unrelated images merged
         into a few giant clusters (MildDemented ended up needing only 221
         clusters to cover 8,513 images). Clustering is now done separately
         within each class.
      2. Splitting by cluster alone (ignoring class) produced wildly
         disproportionate class counts per split (test set had 1,114
         NonDemented but only 79 MildDemented). Each class's clusters are
         now split into train/val/test independently, so every split stays
         class-balanced, and duplicates still never cross a split boundary.
    """
    print("\n===== STEP 4: Leakage-Safe Stratified Split (per-label) =====")
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.utils.class_weight import compute_class_weight

    all_train, all_val, all_test = [], [], []

    for c in CLASSES:
        folder = os.path.join(CLEANED_DST, c)
        files = [os.path.join(folder, img) for img in os.listdir(folder)
                 if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        df_c = pd.DataFrame({'filename': files, 'label': c})
        print(f"\n--- {c}: {len(df_c)} images ---")

        df_c = _hash_all_images(df_c)
        df_c = _cluster_within_label(df_c)

        cluster_sizes = df_c.groupby('cluster').size()
        print(f"{c}: {df_c['cluster'].nunique()} clusters, largest cluster = {cluster_sizes.max()} images")

        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tv_idx, test_idx = next(gss1.split(df_c, groups=df_c['cluster']))
        tv_df, test_df_c = df_c.iloc[tv_idx], df_c.iloc[test_idx]

        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(gss2.split(tv_df, groups=tv_df['cluster']))
        train_df_c, val_df_c = tv_df.iloc[train_idx], tv_df.iloc[val_idx]

        all_train.append(train_df_c)
        all_val.append(val_df_c)
        all_test.append(test_df_c)

    train_df = pd.concat(all_train, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = pd.concat(all_val, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = pd.concat(all_test, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    # --- Balance classes WITHIN each split (safe: leakage is already
    # prevented since each label's clusters were assigned to exactly one
    # split above; undersampling images within one split can't leak into another) ---
    def _balance(split_df, name):
        min_count = split_df['label'].value_counts().min()
        balanced = pd.concat([
            split_df[split_df['label'] == c].sample(n=min_count, random_state=42)
            for c in CLASSES
        ], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"\n{name}: {len(balanced)} images (balanced to {min_count}/class)")
        print(balanced['label'].value_counts())
        return balanced

    train_df = _balance(train_df, "train")
    val_df = _balance(val_df, "val")
    test_df = _balance(test_df, "test")

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_df[['filename', 'label']].to_csv(f"{name}.csv", index=False)

    # Sanity check: no (label, cluster) pair should appear in more than one split
    train_keys = set(zip(train_df['label'], train_df['cluster']))
    val_keys = set(zip(val_df['label'], val_df['cluster']))
    test_keys = set(zip(test_df['label'], test_df['cluster']))
    print("\nCluster overlap check (should all be 0):")
    print(f"  train/val: {len(train_keys & val_keys)}")
    print(f"  train/test: {len(train_keys & test_keys)}")
    print(f"  val/test: {len(val_keys & test_keys)}")

    class_indices = {c: i for i, c in enumerate(CLASSES)}
    y_train_int = train_df['label'].map(class_indices).values
    weights = compute_class_weight(class_weight='balanced', classes=np.arange(len(CLASSES)), y=y_train_int)
    class_weight_dict = {i: float(w) for i, w in enumerate(weights)}

    with open("class_weights.json", "w") as f:
        json.dump(class_weight_dict, f, indent=2)

    print("\nClass weights:", class_weight_dict)
    print("Saved as class_weights.json")


# ---------------- OPTIONAL: STANDALONE LEAKAGE DIAGNOSTIC ----------------
def check_leakage():
    """
    Standalone diagnostic -- run any time after Step 4 to verify no
    near-duplicate leakage exists between the current train.csv and test.csv.
    If your splits came from the current run_split(), this should read 0.
    """
    from PIL import Image
    import imagehash
    from tqdm import tqdm

    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    print("Hashing train images...")
    train_hashes = {}
    for path in tqdm(train_df['filename']):
        try:
            h = imagehash.phash(Image.open(path))
            train_hashes[h] = path
        except Exception:
            continue

    print("Checking test images against train hashes...")
    exact_matches = 0
    near_matches = 0
    train_hash_list = list(train_hashes.keys())

    for path in tqdm(test_df['filename']):
        try:
            h = imagehash.phash(Image.open(path))
        except Exception:
            continue
        if h in train_hashes:
            exact_matches += 1
        else:
            min_dist = min(h - th for th in train_hash_list)
            if min_dist <= HAMMING_THRESHOLD:
                near_matches += 1

    print(f"\nExact duplicate hashes (train vs test): {exact_matches}")
    print(f"Near-duplicate images (hamming distance <= {HAMMING_THRESHOLD}): {near_matches}")
    print(f"Total test set size: {len(test_df)}")
    if exact_matches + near_matches > 0:
        pct = (exact_matches + near_matches) / len(test_df) * 100
        print(f"WARNING: ~{pct:.1f}% of your test set has a near-duplicate in train.")
    else:
        print("No duplicate/near-duplicate images found between train and test. Good sign.")


# ---------------- SHARED TRAINING HELPER (used by Steps 5 & 6) ----------------
def _train_two_phase(model_name, base_model_fn, preprocess_fn, unfreeze_last_n=30):
    """
    Trains a transfer-learning model in two phases:
      Phase 1: backbone frozen, train the classification head.
      Phase 2: unfreeze the top `unfreeze_last_n` layers of the backbone,
               fine-tune everything at a much lower learning rate.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    from sklearn.metrics import classification_report

    with open("class_weights.json") as f:
        class_weight_dict = {int(k): v for k, v in json.load(f).items()}

    train_df = pd.read_csv("train.csv")
    val_df = pd.read_csv("val.csv")
    test_df = pd.read_csv("test.csv")

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
        zoom_range=0.15, brightness_range=[0.85, 1.15],
        horizontal_flip=True, fill_mode='nearest'
    )
    val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    train_gen = train_datagen.flow_from_dataframe(
        train_df, x_col='filename', y_col='label', target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode='categorical', classes=CLASSES
    )
    val_gen = val_test_datagen.flow_from_dataframe(
        val_df, x_col='filename', y_col='label', target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode='categorical', classes=CLASSES, shuffle=False
    )
    test_gen = val_test_datagen.flow_from_dataframe(
        test_df, x_col='filename', y_col='label', target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode='categorical', classes=CLASSES, shuffle=False
    )

    base = base_model_fn(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(4, activation='softmax')(x)
    model = models.Model(inputs=base.input, outputs=outputs)
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

    ckpt_path = os.path.join(MODELS_DIR, f"{model_name}.keras")

    print(f"\n--- {model_name}: Phase 1 (frozen backbone) ---")
    model.fit(
        train_gen, steps_per_epoch=len(train_gen), epochs=EPOCHS_PHASE1,
        validation_data=val_gen, validation_steps=len(val_gen),
        class_weight=class_weight_dict,
        callbacks=[
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.2, patience=5),
            ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_accuracy')
        ]
    )

    print(f"\n--- {model_name}: Phase 2 (fine-tuning top {unfreeze_last_n} layers) ---")
    base.trainable = True
    for layer in base.layers[:-unfreeze_last_n]:
        layer.trainable = False

    model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(
        train_gen, steps_per_epoch=len(train_gen), epochs=EPOCHS_PHASE2,
        validation_data=val_gen, validation_steps=len(val_gen),
        class_weight=class_weight_dict,
        callbacks=[
            EarlyStopping(patience=8, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.2, patience=4),
            ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_accuracy')
        ]
    )

    model.save(ckpt_path)
    print(f"{model_name} saved to {ckpt_path}")

    test_loss, test_acc = model.evaluate(test_gen)
    print(f"\n===== {model_name.upper()} FINAL TEST ACCURACY: {test_acc*100:.2f}% =====")

    preds = np.argmax(model.predict(test_gen), axis=-1)
    print(classification_report(test_gen.classes, preds, target_names=CLASSES))

    return model


# ---------------- STEP 5: TRAIN BASELINE (EfficientNetB0) ----------------
def run_train_baseline():
    print("\n===== STEP 5: Training Baseline EfficientNetB0 (2-phase) =====")
    import tensorflow as tf
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.applications.efficientnet import preprocess_input

    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    if len(tf.config.list_physical_devices('GPU')) == 0:
        print("WARNING: No GPU detected. Strongly consider running this step in Google Colab instead.")

    _train_two_phase("baseline_efficientnet", EfficientNetB0, preprocess_input, unfreeze_last_n=30)


# ---------------- STEP 6: TRAIN OTHER MODELS (FOR ENSEMBLE) ----------------
def run_train_others():
    print("\n===== STEP 6: Training MobileNetV3Large & ResNet50 (2-phase each) =====")
    from tensorflow.keras.applications import MobileNetV3Large, ResNet50
    from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

    _train_two_phase("mobilenet", MobileNetV3Large, mobilenet_preprocess, unfreeze_last_n=20)
    _train_two_phase("resnet50", ResNet50, resnet_preprocess, unfreeze_last_n=20)


# ---------------- STEP 7: ENSEMBLE & EVALUATION ----------------
def run_ensemble():
    print("\n===== STEP 7: Running Ensemble (fixed preprocessing) =====")
    from tensorflow.keras.models import load_model
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
    from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

    test_df = pd.read_csv("test.csv")

    model1 = load_model(os.path.join(MODELS_DIR, "baseline_efficientnet.keras"))
    model2 = load_model(os.path.join(MODELS_DIR, "mobilenet.keras"))
    model3 = load_model(os.path.join(MODELS_DIR, "resnet50.keras"))

    def make_gen(preprocess_fn):
        datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
        return datagen.flow_from_dataframe(
            test_df, x_col='filename', y_col='label', target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE, class_mode='categorical', classes=CLASSES, shuffle=False
        )

    gen1 = make_gen(eff_preprocess)
    gen2 = make_gen(mobilenet_preprocess)
    gen3 = make_gen(resnet_preprocess)

    p1 = model1.predict(gen1)
    p2 = model2.predict(gen2)
    p3 = model3.predict(gen3)

    ensemble_probs = (p1 + p2 + p3) / 3.0
    ensemble_preds = np.argmax(ensemble_probs, axis=-1)
    confidences = np.max(ensemble_probs, axis=-1)
    print(f"Average confidence across test set: {confidences.mean()*100:.2f}%")

    true_labels = gen1.classes

    acc = accuracy_score(true_labels, ensemble_preds)
    print(f"\n===== ENSEMBLE ACCURACY: {acc*100:.2f}% =====")
    print("\nClassification Report:")
    print(classification_report(true_labels, ensemble_preds, target_names=CLASSES))

    cm = confusion_matrix(true_labels, ensemble_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Ensemble Confusion Matrix")
    plt.tight_layout()
    plt.savefig("ensemble_cm.png")
    plt.show()


# ---------------- STEP 8: GRAD-CAM (Explainability) ----------------
def run_gradcam():
    print("\n===== STEP 8: Generating Grad-CAM Explanation =====")
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.efficientnet import preprocess_input
    import cv2

    model = load_model(os.path.join(MODELS_DIR, "baseline_efficientnet.keras"))

    test_df = pd.read_csv("test.csv")
    sample = test_df.sample(1)
    img_path = sample['filename'].values[0]
    label = sample['label'].values[0]

    img = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img)
    img_input = preprocess_input(np.expand_dims(img_array, axis=0))

    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer("top_conv").output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_input)
        pred_class = tf.argmax(predictions[0])
        loss = predictions[:, pred_class]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    img_raw = cv2.imread(img_path)
    img_raw = cv2.resize(img_raw, (IMG_SIZE, IMG_SIZE))
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img_raw, 0.6, heatmap_colored, 0.4, 0)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
    plt.title(f"Original: {label}")
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM (AI Focus)")
    plt.savefig("gradcam_example.png")

    pred_probs = model.predict(img_input)
    pred_class_idx = np.argmax(pred_probs)
    confidence = np.max(pred_probs)
    pred_class_name = CLASSES[pred_class_idx]
    generate_report(pred_class_name, confidence, img_path)
    plt.show()
    print("Grad-CAM saved as 'gradcam_example.png'")


# ---------------- STEP 9: DECISION SUPPORT REPORT ----------------
def generate_report(predicted_class, confidence, img_path=""):
    print("\n" + "=" * 45)
    print("   CLINICAL DECISION SUPPORT REPORT")
    print("=" * 45)
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"Disease Stage: {STAGE_MAP.get(predicted_class, 'Unknown')}")
    print("\nAI Explanation: Highlighted regions (see Grad-CAM) indicate structural")
    print("  patterns associated with this classification.")
    if predicted_class == 'NonDemented':
        print("\nRecommendation: No immediate concern; routine monitoring advised.")
    else:
        print("\nRecommendation: Consult a neurologist for comprehensive clinical evaluation.")
    print("=" * 45)
    return f"Report generated for {os.path.basename(img_path) if img_path else 'image'}"


# ===================================================================
# ====================== CHOOSE YOUR STEP HERE ======================
# ===================================================================
if __name__ == "__main__":
    # STEP 1: Analyze data
    # run_analysis()

    # STEP 2: Clean data (Warning: This deletes bad files)
    # run_cleaning()

    # STEP 3: Merge
    # run_merge()

    # STEP 4: Leakage-safe, stratified, per-label split (re-run — previous split was lopsided)
    #run_split()

    # OPTIONAL: re-verify no leakage remains after Step 4
    # check_leakage()

    # STEP 5: Train Baseline (2-phase: frozen head + fine-tune)
    run_train_baseline()

    # STEP 6: Train Others (2-phase each: MobileNetV3Large + ResNet50)
    # run_train_others()

    # STEP 7: Ensemble (preprocessing bug fixed)
    # run_ensemble()

    # STEP 8: Grad-CAM
    # run_gradcam()