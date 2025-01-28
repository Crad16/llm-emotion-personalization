# scripts/run_mistral.py

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model_utils import load_hf_model, hf_generate_response

MISTRAL_TOKENIZER = None
MISTRAL_MODEL = None

def init_mistral():
    global MISTRAL_TOKENIZER, MISTRAL_MODEL
    if MISTRAL_MODEL is None or MISTRAL_TOKENIZER is None:
        model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        MISTRAL_TOKENIZER, MISTRAL_MODEL = load_hf_model(model_name)

def run_mistral_inference(prompt: str) -> str:
    init_mistral()
    return hf_generate_response(prompt, MISTRAL_TOKENIZER, MISTRAL_MODEL)

def main():
    test_output = run_mistral_inference("Hello from Mistral!")
    print(test_output)

if __name__ == "__main__":
    main()
