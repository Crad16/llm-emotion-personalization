import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_hf_model(model_name: str):
    """
    Loads a Hugging Face model and tokenizer from `model_name`.
    Returns (tokenizer, model).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    return tokenizer, model

def hf_generate_response(prompt: str, tokenizer, model, max_new_tokens=512):
    """
    Generates a text response given a prompt, using a HF-based model.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=50,
        top_p=0.9
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
