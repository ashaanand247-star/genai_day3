import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from config import MODEL_NAME, BASE_URL
from logger import logger

load_dotenv()


class ModelClient:

    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY")

        self.client = OpenAI(
            api_key=api_key,
            base_url=BASE_URL
        )

    def generate(self, system_prompt, user_prompt):

        logger.info("Generating AI response...")

        start_time = time.time()

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
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

        return {
            "text": response.choices[0].message.content,
            "model": response.model,
            "latency": latency,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }