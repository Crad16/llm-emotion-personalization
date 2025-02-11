import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import transformers
import torch

LLAMA_PIPE = None

def init_llama():
    global LLAMA_PIPE
    if LLAMA_PIPE is None:
        model_id = "meta-llama/Llama-3.1-8B-Instruct"
        LLAMA_PIPE = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.float16},
            device_map="auto",
        )

def run_llama_inference(system_prompt: str, user_prompt: str) -> str:
    init_llama()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    outputs = LLAMA_PIPE(messages, max_new_tokens=1000, return_full_text=False)
    return outputs[0]["generated_text"]
