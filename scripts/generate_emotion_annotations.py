import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import pandas as pd

from src.prompts.emotion_prompt import EMOTION_PROMPT_TEMPLATE_SYSTEM, EMOTION_PROMPT_TEMPLATE_USER

from scripts.run_mistral import run_mistral_inference
from scripts.run_gpt4o import run_gpt4o_inference
from scripts.run_qwen import run_qwen_inference
from scripts.run_llama import run_llama_inference

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        choices=["mistral","gpt4o","qwen2.5","llama"],
                        help="Which model to use for emotion annotation")
    parser.add_argument("-id", "--annotator_ids", type=str, required=True,
                        help="Comma-separated list of annotator IDs to process, e.g. '0,1,2'")
    parser.add_argument("-test", "--test_folder", default="data/split_test",
                        help="Folder that contains 'annotator_{id}_test.csv' files")
    parser.add_argument("-o", "--output_folder", default="data/no_personalization",
                        help="Folder to store the output CSV files")
    parser.add_argument("-text", "--text_csv", default="data/original/StudEmo_text_data.csv",
                        help="CSV file mapping text_id to actual text columns")
    parser.add_argument("-s", "--sample_lines", type=int, default=0,
                        help="If > 0, only process the first N lines from the CSV (for quick testing).")
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
        input_csv = os.path.join(args.test_folder, f"annotator_{annot_id}_test.csv")
        if not os.path.exists(input_csv):
            print(f"File not found: {input_csv}")
            continue

        df = pd.read_csv(input_csv)
        if args.sample_lines > 0:
            df = df.head(args.sample_lines)

        print(f"Loaded {len(df)} lines from {input_csv}")

        if "text_id" not in df.columns:
            print(f"No 'text_id' column in {input_csv}, cannot merge with text. Skipping.")
            continue

        merged_df = pd.merge(df, text_df[["text_id", "text"]], on="text_id", how="left")

        outputs = []
        for idx, row in merged_df.iterrows():
            text_content = str(row["text"])
            system_prompt = EMOTION_PROMPT_TEMPLATE_SYSTEM
            user_prompt = EMOTION_PROMPT_TEMPLATE_USER.format(post_text=text_content)

            raw_response = inference_func(system_prompt=system_prompt, user_prompt=user_prompt)
            outputs.append(raw_response)

            if (idx + 1) % 50 == 0:
                print(f"Annotator {annot_id}, processed {idx+1} lines...")

        merged_df["model_annotations"] = outputs

        output_csv = os.path.join(args.output_folder, f"{args.model}_result_{annot_id}.csv")
        merged_df.to_csv(output_csv, index=False)
        print(f"Done! Saved {len(merged_df)} results to {output_csv}")

if __name__ == "__main__":
    main()
