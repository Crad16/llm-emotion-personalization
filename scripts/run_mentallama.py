import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import transformers
import torch

MENTALLAMA_PIPE = None

def init_mentallama():
    global MENTALLAMA_PIPE
    if MENTALLAMA_PIPE is None:
        model_id = "klyang/MentaLLaMA-chat-13B"
        MENTALLAMA_PIPE = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.float16},
            device_map="auto",
        )

def run_mentallama_inference(system_prompt: str, user_prompt: str) -> str:
    init_mentallama()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    outputs = MENTALLAMA_PIPE(messages, max_new_tokens=1000, return_full_text=False)
    return outputs[0]["generated_text"]
