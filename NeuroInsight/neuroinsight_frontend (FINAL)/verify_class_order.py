"""
RUN THIS FIRST, BEFORE USING THE APP.

This checks that CLASS_NAMES in app.py actually matches the class-index
order your model was trained and evaluated with. Getting this wrong means
every prediction in the app would be silently mislabeled (e.g. the app
might say "NonDemented" when the model actually meant "ModerateDemented").

HOW TO USE:
Run this from the same folder where your train_merged.csv / val_merged.csv
/ test_merged.csv live (wherever they ended up after the merge step), with
the same class column name used during training (commonly 'label').
"""

import pandas as pd

try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except ImportError:
    # TensorFlow is required to run this verification script. Importing it inside
    # a guarded block prevents editor warnings when the dependency isn't installed
    # in the current Python environment.
    ImageDataGenerator = None

CSV_PATH = "test_merged.csv"   # <-- point this at your actual merged test CSV
LABEL_COLUMN = "label"          # <-- change if your column is named differently

if ImageDataGenerator is None:
    raise RuntimeError(
        "TensorFlow is required for this script. Install the project dependencies "
        "in the active environment, then rerun this file."
    )

df = pd.read_csv(CSV_PATH)
datagen = ImageDataGenerator()
gen = datagen.flow_from_dataframe(
    df, x_col="filename", y_col=LABEL_COLUMN,
    target_size=(224, 224), batch_size=32, class_mode="categorical"
)

print("\n" + "=" * 60)
print("class_indices as Keras actually assigned them:")
print(gen.class_indices)
print("=" * 60)
print("\nCompare this dict to CLASS_NAMES in app.py.")
print("app.py's CLASS_NAMES list must be ordered so that CLASS_NAMES[i]")
print("matches whichever class name has index i above.")
print("\nExample: if the printed dict shows {'EarlyDemented': 0, 'ModerateDemented': 1, 'NonDemented': 2}")
print("then CLASS_NAMES should be: ['EarlyDemented', 'ModerateDemented', 'NonDemented']")
