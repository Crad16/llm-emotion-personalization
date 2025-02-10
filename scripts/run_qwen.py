import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model_utils import load_hf_model, hf_generate_response

QWEN_TOKENIZER = None
QWEN_MODEL = None

def init_qwen():
    global QWEN_TOKENIZER, QWEN_MODEL
    if QWEN_MODEL is None or QWEN_TOKENIZER is None:
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        QWEN_TOKENIZER, QWEN_MODEL = load_hf_model(model_name)

def run_qwen_inference(system_prompt: str, user_prompt: str) -> str:
    init_qwen()
    return hf_generate_response(system_prompt, user_prompt, QWEN_TOKENIZER, QWEN_MODEL)

def main():
    test_output = run_qwen_inference("Hello from Qwen2.5!")
    print(test_output)

if __name__ == "__main__":
    main()
    