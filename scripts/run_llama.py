import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_utils import load_csv
from src.model_utils import load_hf_model, generate_response

def main():
    df = load_csv("data/StudEmo_text_data.csv")
    # df = df.head(10) # for limited input line
    model_name = "meta-llama/Llama-3.1-8B-Instruct"  # access not granted yet
    tokenizer, model = load_hf_model(model_name)
    model.config.pad_token_id = tokenizer.eos_token_id

    responses = []
    for _, row in df.iterrows():
        text_data = row["text"]
        prompt = f"Default prompt: {text_data}"
        response = generate_response(prompt, tokenizer, model)
        responses.append(response)

    df["llama_response"] = responses
    df.to_csv("data/llama_output.csv", index=False)
    print("Llama generation completed. Results saved to data/llama_output.csv")

if __name__ == "__main__":
    main()
