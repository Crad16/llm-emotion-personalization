import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import transformers
import torch

MISTRAL_PIPE = None

def init_mistral():
    global MISTRAL_PIPE
    if MISTRAL_PIPE is None:
        model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        MISTRAL_PIPE = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.float16},
            device_map="auto"
        )

def run_mistral_inference(system_prompt: str, user_prompt: str) -> str:
    init_mistral()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    outputs = MISTRAL_PIPE(messages, max_new_tokens=1000, return_full_text=False)
    return outputs[0]["generated_text"]
