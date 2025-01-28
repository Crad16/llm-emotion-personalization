import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model_utils import load_hf_model, hf_generate_response

MENTAL_LLAMA_TOKENIZER = None
MENTAL_LLAMA_MODEL = None

def init_mentallama():
    global MENTAL_LLAMA_TOKENIZER, MENTAL_LLAMA_MODEL
    if MENTAL_LLAMA_MODEL is None or MENTAL_LLAMA_TOKENIZER is None:
        model_name = "klyang/MentaLLaMA-chat-13B"
        MENTAL_LLAMA_TOKENIZER, MENTAL_LLAMA_MODEL = load_hf_model(model_name)

def run_mentallama_inference(prompt: str) -> str:
    init_mentallama()
    return hf_generate_response(prompt, MENTAL_LLAMA_TOKENIZER, MENTAL_LLAMA_MODEL)

def main():
    test_output = run_mentallama_inference("Hello from MentaLLaMA!")
    print(test_output)

if __name__ == "__main__":
    main()