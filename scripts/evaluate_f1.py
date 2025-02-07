import json
import argparse
import pandas as pd
from sklearn.metrics import f1_score

LABELS = [
    "joy", "trust", "anticipation", "surprise", "fear",
    "sadness", "disgust", "anger", "valence", "arousal"
]

def binarize(label_name: str, value: float) -> int:
    """
    Converts the numeric `value` into a binary class (0 or 1).
      - For non-valence labels (joy, trust, etc. except valence):
         0 => 0,  any positive => 1
      - For valence:
         < 0 => 1,  else => 0
    """
    if label_name == "valence":
        return 1 if value < 0 else 0
    else:
        # Non-valence
        return 0 if value == 0 else 1

def parse_json_preds(json_str: str):
    # Remove any wrapping triple backticks or '```json' prefix
    # so that only the raw JSON remains
    s = json_str.strip()
    if s.startswith("```json"):
        s = s[7:].strip()  # remove leading ```json
    if s.endswith("```"):
        s = s[:-3].strip() # remove trailing ```
    
    try:
        data = json.loads(s)
        return {lbl: float(data[lbl]) for lbl in LABELS}
    except (json.JSONDecodeError, KeyError, TypeError):
        return None

def evaluate_binary_f1():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=str,
                        help="Evaluating CSV file")
    args = parser.parse_args()
    
    df = pd.read_csv(args.path)

    gt_bins = {lbl: [] for lbl in LABELS}
    pred_bins = {lbl: [] for lbl in LABELS}

    for _, row in df.iterrows():
        try:
            ground_truth_vals = {lbl: float(row[lbl]) for lbl in LABELS}
        except ValueError:
            continue 

        model_data = parse_json_preds(str(row["model_annotations"]))
        if model_data is None:
            continue

        for lbl in LABELS:
            gt_label = binarize(lbl, ground_truth_vals[lbl])
            pred_label = binarize(lbl, model_data[lbl])
            gt_bins[lbl].append(gt_label)
            pred_bins[lbl].append(pred_label)

    f1_scores = {}
    for lbl in LABELS:
        y_true = gt_bins[lbl]
        y_pred = pred_bins[lbl]
        if len(y_true) == 0:
            f1_scores[lbl] = None
        else:
            f1_scores[lbl] = f1_score(y_true, y_pred)

    valid_f1s = [s for s in f1_scores.values() if s is not None]
    if valid_f1s:
        avg_f1 = sum(valid_f1s) / len(valid_f1s)
    else:
        avg_f1 = None

    print("=== Per-Label F1 Scores (Binary) ===")
    for lbl in LABELS:
        val = f1_scores[lbl]
        if val is None:
            print(f"{lbl}: None")
        else:
            print(f"{lbl}: {val:.4f}")
    print("\n=== Average F1 across all 10 labels ===")
    if avg_f1 is None:
        print("Average F1: None")
    else:
        print(f"Average F1: {avg_f1:.4f}")

if __name__ == "__main__":
    evaluate_binary_f1()
