import functools
import json
from typing import Any, Literal, Optional, cast

import backoff
import openai
import openai.error
from openai.openai_object import OpenAIObject

from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory, cache_turn_on
from dsp.modules.lm import SelfLM


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
        model: str = "gpt-3.5-turbo",
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

        if api_key:
            openai.api_key = api_key

        self.kwargs = {
            "temperature": 0.0,
            "max_tokens": 300,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
            **kwargs,
        }
        if api_provider == "openai":
            self.kwargs["model"] = model
        self.history: list[dict[str, Any]] = []

    def basic_request(self, prompt: str, **kwargs) -> OpenAIObject:
        raw_kwargs = kwargs
        kwargs = {**self.kwargs, **kwargs}
        # Always use the chat format
        kwargs["messages"] = [{"role": "user", "content": prompt}]
        # Stringify the request for caching purposes
        kwargs = {"stringify_request": json.dumps(kwargs)}
        response = cached_chat_request(**kwargs)
        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)
        return response

    @backoff.on_exception(
        backoff.expo,
        (openai.error.RateLimitError, openai.error.ServiceUnavailableError),
        max_time=1000,
        on_backoff=backoff_hdlr,
    )
    def request(self, prompt: str, **kwargs) -> OpenAIObject:
        return self.basic_request(prompt, **kwargs)

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
        choices = response["choices"]
        completed_choices = [c for c in choices if c.get("finish_reason", "") != "length"]
        if only_completed and completed_choices:
            choices = completed_choices
        completions = [c["message"]["content"] for c in choices]
        # Note: Sorting by logprobs is not applicable to ChatCompletion responses.
        return completions


@CacheMemory.cache
def cached_chat_request_v2(**kwargs) -> OpenAIObject:
    if "stringify_request" in kwargs:
        kwargs = json.loads(kwargs["stringify_request"])
    return cast(OpenAIObject, openai.ChatCompletion.create(**kwargs))


@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@NotebookCacheMemory.cache
def cached_chat_request_v2_wrapped(**kwargs) -> OpenAIObject:
    return cached_chat_request_v2(**kwargs)


cached_chat_request = cached_chat_request_v2_wrapped
