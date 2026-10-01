# NeuroInsight — Local Frontend

A local web app that serves your trained 3-class Alzheimer's MRI model:
upload a scan, get a classification, a Grad-CAM overlay showing what the
model focused on, and a decision support report — all in the browser.

## 1. Verify your class order FIRST (do not skip this)

Wrong class ordering means every prediction is silently mislabeled. Before
anything else:

1. Copy `verify_class_order.py` into the same folder as your
   `test_merged.csv` (or wherever your merged 3-class CSVs ended up).
2. Edit the `CSV_PATH` and `LABEL_COLUMN` variables at the top if needed.
3. Run it:
   ```
   python verify_class_order.py
   ```
4. It prints something like:
   ```
   {'EarlyDemented': 0, 'ModerateDemented': 1, 'NonDemented': 2}
   ```
5. Open `app.py` and make sure `CLASS_NAMES` is ordered to match — index 0
   in the list must be whichever class shows index 0 above, and so on.

## 2. Add your model files

Place these two files into the `models/` folder:
- `resnet50_finetuned.keras` (your fine-tuned 3-class model)
- `optimal_threshold.pkl` (your tuned NonDemented threshold — optional; if
  missing, the app falls back to the hardcoded `NONDEMENTED_THRESHOLD = 0.3232`
  in `app.py`)

If your files have different names, update `MODEL_PATH` / `THRESHOLD_PATH`
at the top of `app.py` to match.

## 3. Check the threshold logic matches what you actually did

`app.py`'s `apply_threshold()` function assumes your threshold tuning
worked like this: "NonDemented only wins if its own predicted probability
clears the threshold; otherwise pick the highest of the other two classes."
This is the most common way this kind of fix is implemented, but if your
actual code (or DeepSeek's) did something different, edit that function to
match — otherwise this app's predictions won't line up with the accuracy
numbers in your report.

## 4. Install dependencies

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # Windows
# source venv/bin/activate       # Mac/Linux
pip install -r requirements.txt
```

## 5. Run it

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

## Project structure

```
neuroinsight_frontend/
  app.py                    # Flask backend: model, Grad-CAM, report generation
  verify_class_order.py     # RUN THIS FIRST
  requirements.txt
  models/
    resnet50_finetuned.keras   # <- you add this
    optimal_threshold.pkl      # <- you add this (optional)
  templates/
    index.html
  static/
    css/style.css
    js/main.js
```

## Notes on what this app does NOT do

- It does not use test-time augmentation (TTA) for single-image predictions.
  Your own results showed TTA actually *reduced* accuracy on your test set
  (88.75% vs. 89.88% without it) — so the app intentionally skips it for
  faster, more accurate single predictions.
- It reports 3 classes (Early / Moderate / Non-Demented), matching your
  validated ~90% model — not the original 4-class breakdown, which scored
  lower (~82%) and was superseded by the merge.
- The disclaimer banner is not decorative — keep it if you present or
  submit this project. A tool that predicts a medical condition from an
  image should always be clear about what it is and isn't.
