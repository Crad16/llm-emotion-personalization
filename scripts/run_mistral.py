import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_utils import load_csv
from src.model_utils import load_hf_model, generate_response

def main():
    df = load_csv("data/StudEmo_text_data.csv")

    model_name = "mistralai/Mistral-7B-Instruct-v0.3"  
    tokenizer, model = load_hf_model(model_name)

    responses = []
    for _, row in df.iterrows():
        text_data = row["text"]
        prompt = f"Default prompt: {text_data}"
        response = generate_response(prompt, tokenizer, model)
        responses.append(response)

    df["mistral_response"] = responses
    df.to_csv("data/mistral_output.csv", index=False)
    print("Mistral generation completed. Results saved to data/mistral_output.csv")

if __name__ == "__main__":
    main()
