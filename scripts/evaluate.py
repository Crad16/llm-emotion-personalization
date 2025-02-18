import json
import argparse
import pandas as pd
from sklearn.metrics import f1_score

LABELS = [
    "joy", "trust", "anticipation", "surprise", "fear",
    "sadness", "disgust", "anger", "valence", "arousal"
]

def binarize(label_name: str, value: float) -> int:
    if label_name == "valence":
        return 0 if value < 0 else 1
    else:
        return 0 if value == 0 else 1

def parse_json_preds(json_str: str):
    s = json_str.strip()
    if s.startswith("```json"):
        s = s[7:].strip()
    if s.endswith("```"):
        s = s[:-3].strip()
    if s.startswith("(") and s.endswith(")"):
        s = "{" + s[1:-1] + "}"

    try:
        data = json.loads(s)
        return {lbl: float(data[lbl]) for lbl in LABELS}
    except (json.JSONDecodeError, KeyError, TypeError):
        return None

def evaluate_single_file(path: str, skip_lines: int = 0):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"File not found: {path}")
        return None

    if skip_lines >= len(df):
        print(f"Skip lines {skip_lines} >= file length {len(df)}, no data left.")
        return None
    df = df.iloc[skip_lines:].copy()

    missing_labels = [lbl for lbl in LABELS if lbl not in df.columns]
    if missing_labels:
        print(f"Missing ground-truth columns: {missing_labels} in {path}")
        return None
    if "model_annotations" not in df.columns:
        print(f"No 'model_annotations' column in {path}.")
        return None

    gt_bins = {lbl: [] for lbl in LABELS}
    pred_bins = {lbl: [] for lbl in LABELS}
    row_count = 0

    for _, row in df.iterrows():
        try:
            ground_truth_vals = {lbl: float(row[lbl]) for lbl in LABELS}
        except ValueError:
            continue
        model_data = parse_json_preds(str(row["model_annotations"]))
        if model_data is None:
            continue

        row_count += 1
        for lbl in LABELS:
            gt_label = binarize(lbl, ground_truth_vals[lbl])
            pred_label = binarize(lbl, model_data[lbl])
            gt_bins[lbl].append(gt_label)
            pred_bins[lbl].append(pred_label)

    if row_count == 0:
        print("No valid rows to evaluate.")
        return None

    results = {}
    for lbl in LABELS:
        y_true = gt_bins[lbl]
        y_pred = pred_bins[lbl]
        if not y_true:
            results[f"{lbl}_f1"] = None
            results[f"{lbl}_acc"] = None
            continue
        f1_val = f1_score(y_true, y_pred)
        correct = sum(a == b for a,b in zip(y_true,y_pred))
        acc_val = correct / len(y_true)
        results[f"{lbl}_f1"] = f1_val
        results[f"{lbl}_acc"] = acc_val

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True,
                        help="Path to a single CSV file with columns [joy..arousal, model_annotations].")
    parser.add_argument("--skip_lines", type=int, default=0,
                        help="Number of lines to skip (excluding header).")
    args = parser.parse_args()

    results = evaluate_single_file(args.path, skip_lines=args.skip_lines)
    if results is None:
        print("No results computed.")
        return

    f1_vals = [v for k,v in results.items() if k.endswith("_f1") and v is not None]
    acc_vals = [v for k,v in results.items() if k.endswith("_acc") and v is not None]
    avg_f1 = sum(f1_vals)/len(f1_vals) if f1_vals else None
    avg_acc = sum(acc_vals)/len(acc_vals) if acc_vals else None

    print(f"Results for file: {args.path}")
    print(f"Skip lines: {args.skip_lines}")
    for lbl in LABELS:
        f1_key = f"{lbl}_f1"
        acc_key = f"{lbl}_acc"
        f1_sc = results[f1_key] if f1_key in results else None
        acc_sc = results[acc_key] if acc_key in results else None
        print(f"{lbl}_f1: {f1_sc:.4f}" if f1_sc is not None else f"{lbl}_f1: None",
              f"{lbl}_acc: {acc_sc:.4f}" if acc_sc is not None else f"{lbl}_acc: None")
    print(f"avg_f1: {avg_f1:.4f}" if avg_f1 is not None else "avg_f1: None")
    print(f"avg_acc: {avg_acc:.4f}" if avg_acc is not None else "avg_acc: None")

if __name__ == "__main__":
    main()
