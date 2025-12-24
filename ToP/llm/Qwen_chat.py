import openai
import os
from typing import List, Union, Optional
from dotenv import load_dotenv
from dataclasses import asdict
import random
from openai import OpenAI, AsyncOpenAI
import async_timeout
from tenacity import retry, wait_random_exponential, stop_after_attempt
import asyncio
from agent.llm.llm_registry import LLMRegistry,LLM
from agent.llm.format import LLMMessage

load_dotenv()
#logger = get_logger()


BASE_URL= os.getenv("BASE_URL")
API_KEY=os.getenv("API_KEY")

@LLMRegistry.register('QwenChat')
class Qwebnhat(LLM):
    def __init__(self, model_name: str = "Qwen2.5-32B-Instruct-AWQ"):
        self.model_name = model_name

    def generate_response(self, messages: List[LLMMessage], max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE


        client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
        formatted_messages = [asdict(message) for message in messages]
        response = client.chat.completions.create(
            model=self.model_name,
            messages=formatted_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content

    @retry(wait=wait_random_exponential(max=60), stop=stop_after_attempt(5))
    async def generate_response_async(self, messages: List[LLMMessage], max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        #logger.log_prompt(messages)
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        aclient = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

        formatted_messages = [asdict(message) for message in messages]
        try:
            async with async_timeout.timeout(100):
                response = await aclient.chat.completions.create(
                    model=self.model_name,
                    messages=formatted_messages,
                    # max_tokens=max_tokens,
                    temperature=temperature,
                )
        except asyncio.TimeoutError:
            raise TimeoutError("Timeout")

        return response.choices[0].message.content
