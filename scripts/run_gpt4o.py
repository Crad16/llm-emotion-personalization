import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import openai

openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_KEY_HERE")

def run_gpt4o_inference(prompt: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4o-2024-05-13",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=512
    )
    return response["choices"][0]["message"]["content"]

def main():
    test_output = run_gpt4o_inference("Hello from GPT-4o!")
    print(test_output)

if __name__ == "__main__":
    main()