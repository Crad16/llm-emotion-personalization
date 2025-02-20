import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def run_gpt4o_inference(system_prompt: str, user_prompt: str) -> str:
    response = openai.OpenAI(api_key=openai.api_key).chat.completions.create(
        model="gpt-4o-2024-05-13",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content 