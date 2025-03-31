from abc import ABC, abstractmethod


class SelfLM(ABC):
    """
    Abstract class for language models.

    This base class now assumes an OpenAI-style provider and supports chat-based models,
    including gpt-3.5-turbo and gpt-4. Subclasses should implement basic_request() and __call__().
    """

    def __init__(self, model: str):
        self.kwargs = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 150,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
        }
        # Set the provider to openai for chat completions (GPT-3.5-turbo, GPT-4, etc.)
        self.provider = "openai"
        self.history = []

    @abstractmethod
    def basic_request(self, prompt, **kwargs):
        """
        Abstract method to issue a basic request to the language model.
        """
        pass

    def request(self, prompt, **kwargs):
        return self.basic_request(prompt, **kwargs)

    def print_green(self, text: str, end: str = "\n"):
        print("\x1b[32m" + str(text) + "\x1b[0m", end=end)

    def print_red(self, text: str, end: str = "\n"):
        print("\x1b[31m" + str(text) + "\x1b[0m", end=end)

    def inspect_history(self, n: int = 1):
        """
        Prints the last n prompts and their completions.
        """
        last_prompt = None
        printed = []
        # Iterate over the last 100 history items
        for entry in reversed(self.history[-100:]):
            prompt = entry["prompt"]
            if prompt != last_prompt:
                # For openai provider, choices are stored in entry["response"]["choices"]
                choices = entry["response"]["choices"]
                printed.append((prompt, choices))
            last_prompt = prompt
            if len(printed) >= n:
                break

        for prompt, choices in reversed(printed):
            print("\n\n\n")
            print(prompt, end="")
            text = ""
            if self.provider == "openai":
                text = self._get_choice_text(choices[0])
            else:
                text = choices[0].get("text", "")
            self.print_green(text, end="")
            if len(choices) > 1:
                self.print_red(f" \t (and {len(choices) - 1} other completions)", end="")
            print("\n\n\n")

    def _get_choice_text(self, choice: dict) -> str:
        """
        Default method to extract text from a completion choice.
        For chat models (gpt-3.5-turbo, gpt-4), it returns the content of the message.
        """
        if "message" in choice:
            return choice["message"].get("content", "")
        return choice.get("text", "")

    @abstractmethod
    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        """
        Abstract method to retrieve completions from the language model.
        """
        pass
