import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import transformers
import torch

QWEN_PIPE = None

def init_qwen():
    global QWEN_PIPE
    if QWEN_PIPE is None:
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        QWEN_PIPE = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.float16},
            device_map="auto",
        )

def run_qwen_inference(system_prompt: str, user_prompt: str) -> str:
    init_qwen()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    outputs = QWEN_PIPE(messages, max_new_tokens=1000, return_full_text=False)
    return outputs[0]["generated_text"]
