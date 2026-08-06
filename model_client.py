import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"


class ModelClient:

    def __init__(self):
        self.client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL
        )

    def generate(self, system_prompt, user_prompt):

        start_time = time.time()

        response = self.client.chat.completions.create(
    model="nvidia/nemotron-3-nano-30b-a3b:free",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )

        end_time = time.time()

        latency = end_time - start_time

        generated_text = response.choices[0].message.content
        model_name = response.model

        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens

        return {
            "text": generated_text,
            "model": model_name,
            "latency": latency,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }