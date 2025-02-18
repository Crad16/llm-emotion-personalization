import json
import argparse
import os
import pandas as pd
from sklearn.metrics import f1_score

LABELS = [
    "joy", "trust", "anticipation", "surprise", "fear",
    "sadness", "disgust", "anger", "valence", "arousal"
]

ANNOT_IDS = [1,7,10,11,12,18,19,20,23,24]

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

def evaluate_file(path: str, skip_lines=0):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"File not found: {path}")
        return None

    if skip_lines > 0:
        if skip_lines >= len(df):
            print(f"Skip lines {skip_lines} >= file length {len(df)}, no data left in {path}")
            return None
        df = df.iloc[skip_lines:].copy()

    if not all(lbl in df.columns for lbl in LABELS):
        print(f"Missing ground-truth columns in {path}.")
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
        print(f"No valid rows in {path}.")
        return None

    results = {}
    for lbl in LABELS:
        y_true = gt_bins[lbl]
        y_pred = pred_bins[lbl]
        if len(y_true) == 0:
            results[f"{lbl}_f1"] = None
            results[f"{lbl}_acc"] = None
            continue
        # F1
        f1_val = f1_score(y_true, y_pred)
        # Accuracy
        correct = sum(a == b for a,b in zip(y_true,y_pred))
        acc_val = correct / len(y_true)
        results[f"{lbl}_f1"] = f1_val
        results[f"{lbl}_acc"] = acc_val

    return results

def build_rows_for_case(case_name: str, file_pattern: str, skip_lines=0):
    rows = []
    for annot_id in ANNOT_IDS:
        path = file_pattern.format(annot_id=annot_id)
        print(f"Evaluating {path} with skip_lines={skip_lines} ...")
        scores_dict = evaluate_file(path, skip_lines=skip_lines)
        if scores_dict is None:
            row = {"type": case_name, "annotator_id": annot_id}
            for lbl in LABELS:
                row[f"{lbl}_f1"] = None
                row[f"{lbl}_acc"] = None
            row["avg_f1"] = None
            row["avg_acc"] = None
            rows.append(row)
            continue

        # average across labels
        f1_vals = [v for k,v in scores_dict.items() if k.endswith("_f1") and v is not None]
        acc_vals = [v for k,v in scores_dict.items() if k.endswith("_acc") and v is not None]
        avg_f1 = sum(f1_vals)/len(f1_vals) if f1_vals else None
        avg_acc = sum(acc_vals)/len(acc_vals) if acc_vals else None

        row = {"type": case_name, "annotator_id": annot_id}
        row.update(scores_dict)
        row["avg_f1"] = avg_f1
        row["avg_acc"] = avg_acc
        rows.append(row)

    # summary row for that case
    summary_dict = {"type": case_name, "annotator_id": "Summary"}
    for lbl in LABELS:
        lbl_f1_vals = [r[f"{lbl}_f1"] for r in rows if r[f"{lbl}_f1"] is not None]
        lbl_acc_vals = [r[f"{lbl}_acc"] for r in rows if r[f"{lbl}_acc"] is not None]
        summary_dict[f"{lbl}_f1"] = sum(lbl_f1_vals)/len(lbl_f1_vals) if lbl_f1_vals else None
        summary_dict[f"{lbl}_acc"] = sum(lbl_acc_vals)/len(lbl_acc_vals) if lbl_acc_vals else None

    all_avg_f1 = [r["avg_f1"] for r in rows if r["avg_f1"] is not None and r["annotator_id"] != "Summary"]
    all_avg_acc = [r["avg_acc"] for r in rows if r["avg_acc"] is not None and r["annotator_id"] != "Summary"]
    summary_dict["avg_f1"] = sum(all_avg_f1)/len(all_avg_f1) if all_avg_f1 else None
    summary_dict["avg_acc"] = sum(all_avg_acc)/len(all_avg_acc) if all_avg_acc else None

    rows.append(summary_dict)
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["mistral","gpt4o","qwen2.5","llama"],
                        help="Model name to evaluate (used in file patterns).")
    args = parser.parse_args()

    all_rows = []

    # 1) non-RAG
    non_rag_pattern = f"data/no_personalization/{args.model}_result_{{annot_id}}.csv"
    all_rows.extend(build_rows_for_case("non-RAG", non_rag_pattern, skip_lines=0))

    # 2) RAG size=50, k=4
    rag_50_pattern = f"data/personalization/{args.model}_rag_50_4_result_{{annot_id}}.csv"
    all_rows.extend(build_rows_for_case("RAG_50_4", rag_50_pattern, skip_lines=0))

    # 3) non-RAG skip lines 150 (to set the same test set)
    all_rows.extend(build_rows_for_case("non-RAG-s150", non_rag_pattern, skip_lines=150))

    # 4) RAG size=200, k=4
    rag_200_pattern = f"data/personalization/{args.model}_rag_200_4_result_{{annot_id}}.csv"
    all_rows.extend(build_rows_for_case("RAG_200_4", rag_200_pattern, skip_lines=0))

    # columns
    columns = ["type", "annotator_id"]
    for lbl in LABELS:
        columns.append(f"{lbl}_f1")
        columns.append(f"{lbl}_acc")
    columns.append("avg_f1")
    columns.append("avg_acc")

    df = pd.DataFrame(all_rows, columns=columns)

    # format numeric
    for col in columns:
        if col not in ["type","annotator_id"]:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if x is not None else None)

    print("\n=== Final Results Table ===")
    print(df)

    os.makedirs(os.path.dirname(f"scores/{args.model}.csv"), exist_ok=True)
    df.to_csv(f"scores/{args.model}.csv", index=False)
    print(f"\nSaved results to scores/{args.model}.csv")

if __name__ == "__main__":
    main()
