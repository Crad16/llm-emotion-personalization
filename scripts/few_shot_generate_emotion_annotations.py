import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import pandas as pd

from src.prompts.few_shot_emotion_prompt import (
    FEW_SHOT_EMOTION_PROMPT_TEMPLATE_SYSTEM,
    FEW_SHOT_EMOTION_PROMPT_TEMPLATE_USER
)

from scripts.run_mistral import run_mistral_inference
from scripts.run_gpt4o import run_gpt4o_inference
from scripts.run_qwen import run_qwen_inference
from scripts.run_llama import run_llama_inference
from scripts.run_mentallama import run_mentallama_inference

LABELS = [
    "joy", "trust", "anticipation", "surprise", "fear",
    "sadness", "disgust", "anger", "valence", "arousal"
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str,
                        choices=["mistral","gpt4o","qwen2.5","llama","mental_llama"],
                        help="Which model to use for emotion annotation")
    parser.add_argument("--annotator_ids", type=str, required=True,
                        help="Comma-separated list of annotator IDs (e.g. '0,1,2')")
    parser.add_argument("--split_folder", default="data/annotator_split",
                        help="Folder that contains 'annotation_data_annotator_{id}.csv' files")
    parser.add_argument("--output_folder", default="data/no_personalization",
                        help="Folder to store the output CSV files")
    parser.add_argument("--text_csv", default="data/original/StudEmo_text_data.csv",
                        help="CSV file mapping text_id to actual text columns")
    parser.add_argument("--skip_lines", type=int, default=50,
                        help="Total lines from the top (excluding header) that won't be used for final testing.\nWithin these lines, the first 'few_shot_size' lines become few-shot examples.")
    parser.add_argument("--few_shot_size", type=int, default=4,
                        help="Number of lines (within skip_lines) to treat as few-shot examples.")
    args = parser.parse_args()

    if args.model == "mistral":
        inference_func = run_mistral_inference
    elif args.model == "gpt4o":
        inference_func = run_gpt4o_inference
    elif args.model == "qwen2.5":
        inference_func = run_qwen_inference
    elif args.model == "llama":
        inference_func = run_llama_inference
    elif args.model == "mental_llama":
        inference_func = run_mentallama_inference
    else:
        raise ValueError(f"Unknown model: {args.model}")

    annotator_ids = [int(a.strip()) for a in args.annotator_ids.split(",")]

    text_df = pd.read_csv(args.text_csv)
    if "text_id" not in text_df.columns or "text" not in text_df.columns:
        raise ValueError("text_csv must have at least 'text_id' and 'text' columns.")

    os.makedirs(args.output_folder, exist_ok=True)

    for annot_id in annotator_ids:
        input_csv = os.path.join(args.split_folder, f"annotation_data_annotator_{annot_id}.csv")
        if not os.path.exists(input_csv):
            print(f"File not found: {input_csv}")
            continue

        df = pd.read_csv(input_csv)
        print(f"Loaded {len(df)} lines from {input_csv}")

        if "text_id" not in df.columns:
            print(f"No 'text_id' column in {input_csv}, cannot merge with text. Skipping.")
            continue

        top_part = df.head(args.skip_lines)
        few_shot_df = top_part.head(args.few_shot_size)
        test_df = df.iloc[args.skip_lines:].copy()

        print(f"Annotator {annot_id}: skip_lines={args.skip_lines}, few_shot_size={args.few_shot_size}")
        print(f"  => few_shot_df = {len(few_shot_df)} lines, skipped portion = {len(top_part)-len(few_shot_df)} lines, test_df = {len(test_df)} lines")

        merged_fewshot = pd.merge(few_shot_df, text_df[["text_id","text"]], on="text_id", how="left")

        few_shot_examples = []
        for _, row_few in merged_fewshot.iterrows():
            ex_text = str(row_few["text"])
            ann_info = ", ".join([f"{lbl}={row_few[lbl]}" for lbl in LABELS if lbl in row_few])
            few_shot_examples.append(f"Text: {ex_text}\nAnnotations: {ann_info}\n")

        few_shot_context = "\n".join(few_shot_examples)

        merged_test = pd.merge(test_df, text_df[["text_id", "text"]], on="text_id", how="left")

        outputs = []
        for idx, row_test in merged_test.iterrows():
            post_text = str(row_test["text"])

            system_prompt = FEW_SHOT_EMOTION_PROMPT_TEMPLATE_SYSTEM
            user_prompt = FEW_SHOT_EMOTION_PROMPT_TEMPLATE_USER.format(
                few_shot_context=few_shot_context,
                post_text=post_text
            )

            raw_response = inference_func(system_prompt=system_prompt, user_prompt=user_prompt)
            outputs.append(raw_response)

            if (idx + 1) % 50 == 0:
                print(f"Annotator {annot_id}, processed {idx+1} lines...")

        merged_test["model_annotations"] = outputs

        output_csv = os.path.join(args.output_folder, f"{args.model}_{args.skip_lines}_{args.few_shot_size}_result_{annot_id}.csv")
        merged_test.to_csv(output_csv, index=False)
        print(f"Done! Saved {len(merged_test)} results to {output_csv}")


if __name__ == "__main__":
    main()
