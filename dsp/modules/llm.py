import functools
import json
from typing import Any, Literal, Optional, cast

import backoff
import openai
# import openai.exceptions
# import openai.error
# from openai.openai_object import OpenAIObject

from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory, cache_turn_on
from dsp.modules.lm import SelfLM

model_pricing = {
    "gpt-4o-mini": {"input": 0.150 / 1e6, "output": 0.600 / 1e6},
    "gpt-4o": {"input": 2.50 / 1e6, "output": 10.00 / 1e6},
    "gpt-4-turbo": {"input": 10.00 / 1e6, "output": 30.00 / 1e6},
    "gpt-4-turbo-preview": {"input": 10.00 / 1e6, "output": 30.00 / 1e6},
    "gpt-4-0125-preview": {"input": 10.00 / 1e6, "output": 30.00 / 1e6},
    "gpt-4-1106-preview": {"input": 10.00 / 1e6, "output": 30.00 / 1e6},
    "gpt-4": {"input": 30.00 / 1e6, "output": 60.00 / 1e6},
    "gpt-4-32k": {"input": 60.00 / 1e6, "output": 120.00 / 1e6},
    "gpt-3.5-turbo-1106": {"input": 1.00 / 1e6, "output": 2.00 / 1e6},
    "gpt-3.5-turbo": {"input": 0.50 / 1e6, "output": 1.50 / 1e6},
    "gpt-3.5-turbo-16k": {"input": 3.00 / 1e6, "output": 4.00 / 1e6},
    "gpt-3.5-turbo-0125": {"input": 0.50 / 1e6, "output": 1.50 / 1e6},
    "gpt-3.5-turbo-instruct": {"input": 1.50 / 1e6, "output": 2.00 / 1e6},
    "o1": {"input": 15.00 / 1e6, "output": 60.00 / 1e6},
    "o1-preview": {"input": 15.00 / 1e6, "output": 60.00 / 1e6},
    "o1-2024-12-17": {"input": 15.00 / 1e6, "output": 60.00 / 1e6},
    "o3-mini": {"input": 1.10 / 1e6, "output": 4.40 / 1e6},
    "o3-mini-2025-01-31": {"input": 1.10 / 1e6, "output": 4.40 / 1e6},
}

def backoff_hdlr(details):
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs {kwargs}".format(**details)
    )


class SelfLLM(SelfLM):
    """
    Wrapper around OpenAI's Chat API using the latest models (e.g. gpt-3.5-turbo).
    This version always uses the ChatCompletion endpoint, converting the prompt to a list of messages.
    
    Args:
        model (str, optional): Chat model to use. Defaults to "gpt-3.5-turbo".
        api_key (Optional[str], optional): API provider authentication token. Defaults to None.
        api_provider (Literal["openai", "azure"], optional): The API provider to use. Defaults to "openai".
        **kwargs: Additional arguments passed to the API.
    """
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        api_provider: Literal["openai", "azure"] = "openai",
        **kwargs,
    ):
        super().__init__(model)
        self.provider = "openai"
        if api_provider == "azure":
            assert (
                "engine" in kwargs or "deployment_id" in kwargs
            ), "Must specify engine or deployment_id for Azure API instead of model."
            assert "api_version" in kwargs, "Must specify api_version for Azure API"
            assert "api_base" in kwargs, "Must specify api_base for Azure API"
            openai.api_type = "azure"
            openai.api_base = kwargs["api_base"]
            if kwargs.get("api_version"):
                openai.api_version = kwargs["api_version"]

        # if api_key:
        #     openai.api_key = api_key
        self.kwargs = {
            "temperature": 0.0,
            "max_tokens": 300,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
            **kwargs,
        }
        
        self.client = None
        if api_provider == "openai" and api_key:
            self.kwargs["model"] = model
            self.client = openai.OpenAI(api_key=api_key)
        
        self.history: list[dict[str, Any]] = []
        self.api_cost = 0.0

    def basic_request(self, prompt: str, **kwargs) -> Any:
        raw_kwargs = kwargs
        kwargs = {**self.kwargs, **kwargs}
        # Always use the chat format
        kwargs["messages"] = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(**kwargs)
        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)
        
        if response.usage:
            cost = self.calculate_cost(
                response.usage.prompt_tokens, 
                response.usage.completion_tokens
            )
            self.api_cost += cost
        #     print(f"[Cost Tracking] This call cost ${cost:.6f}. Total cost so far: ${self.api_cost:.6f}")
        # else:
        #     print("[Cost Tracking] No usage data found in the response.")
        return response

    @backoff.on_exception(
        backoff.expo,
        (openai._exceptions.RateLimitError, openai._exceptions.APIError),
        max_time=1000,
        on_backoff=backoff_hdlr,
    )
    def request(self, prompt: str, **kwargs) -> Any:
        return self.basic_request(prompt, **kwargs)

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = model_pricing.get(self.kwargs["model"], model_pricing["gpt-4o"])
        input_cost = input_tokens * pricing["input"]
        output_cost = output_tokens * pricing["output"]
        return input_cost + output_cost
    
    def show_cost(self) -> float:
        print(f"[Cost Tracking] Total cost so far: ${self.api_cost:.6f}")
           
    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[str]:
        """
        Retrieves completions from the chat model.
        
        Args:
            prompt (str): Prompt to send.
            only_completed (bool, optional): Return only completed responses. Defaults to True.
            return_sorted (bool, optional): Sorting not applicable for chat completions. Defaults to False.
        
        Returns:
            list[str]: List of completion strings.
        """
        response = self.request(prompt, **kwargs)
        choices = response.choices
        completed_choices = [c for c in choices if c.finish_reason != "length"]
        if only_completed and completed_choices:
            choices = completed_choices
        completions = [c.message.content for c in choices]
        # Note: Sorting by logprobs is not applicable to ChatCompletion responses.
        return completions
