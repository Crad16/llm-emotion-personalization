import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import pandas as pd
import json

# Import the prompt template
from src.prompts.emotion_prompt import EMOTION_PROMPT_TEMPLATE

# Import each model's inference function
from scripts.run_mistral import run_mistral_inference
from scripts.run_gpt4o import run_gpt4o_inference
from scripts.run_qwen import run_qwen_inference
from scripts.run_llama import run_llama_inference
from scripts.run_mentallama import run_mentallama_inference

def main():
    # Parse command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str,
                        choices=["mistral","gpt4o","qwen2.5","llama","mental_llama"],
                        help="Which model to use for emotion annotation")
    parser.add_argument("--input_csv", default="data/StudEmo_text_data.csv",
                        help="Path to input CSV with 'text' column")
    parser.add_argument("--output_csv", default="data/emotion_annotated_results.csv",
                        help="Where to save results")
    parser.add_argument("--start_line", type=int, default=0,
                        help="Index of the first row to process (0-based, inclusive)")
    parser.add_argument("--end_line", type=int, default=None,
                        help="Index of the last row to process (exclusive). If not set, process until the end.")
    args = parser.parse_args()

    # Assign the correct inference function
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

    # Load data
    df = pd.read_csv(args.input_csv)

    df = df.iloc[args.start_line:args.end_line]

    # Generate the annotations
    outputs = []
    for idx, row in df.iterrows():
        text = row["text"]

        prompt = EMOTION_PROMPT_TEMPLATE.format(post_text=text)

        raw_response = inference_func(prompt)
        json_str = extract_json_from_response(raw_response)
        outputs.append(json_str)

        if idx % 50 == 0:
            print(f"Processed row {idx}...")

    # Save results
    df[f"{args.model}_annotations"] = outputs
    df.to_csv(args.output_csv, index=False)
    print(f"Done! Results saved to {args.output_csv}")
    

def extract_json_from_response(response: str) -> str:
    """
    Locates the first '{' and the last '}' in `response`
    and returns that substring as the extracted JSON string.
    If parsing fails, returns the raw response.
    """
    try:
        start_index = response.index('{')
        end_index = response.rindex('}') + 1
        json_str = response[start_index:end_index]
        return json_str
    except ValueError:
        # If '{' or '}' isn't found
        return response


if __name__ == "__main__":
    main()
