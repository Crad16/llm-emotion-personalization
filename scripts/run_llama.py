import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model_utils import load_hf_model, hf_generate_response

LLAMA_TOKENIZER = None
LLAMA_MODEL = None

def init_llama():
    global LLAMA_TOKENIZER, LLAMA_MODEL
    if LLAMA_MODEL is None or LLAMA_TOKENIZER is None:
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        LLAMA_TOKENIZER, LLAMA_MODEL = load_hf_model(model_name)

def run_llama_inference(system_prompt: str, user_prompt: str) -> str:
    init_llama()
    return hf_generate_response(system_prompt, user_prompt, LLAMA_TOKENIZER, LLAMA_MODEL)

def main():
    test_output = run_llama_inference("Hello from LLaMA!")
    print(test_output)

if __name__ == "__main__":
    main()
