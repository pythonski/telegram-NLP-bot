import logging
from abc import ABC, abstractmethod
from typing import Any
import backoff
import groq
from groq import Groq


class LM(ABC):
    """Abstract class for language models."""

    def __init__(self, model):
        self.kwargs = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 150,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
        }
        self.provider = "default"
        self.history = []

    @abstractmethod
    def basic_request(self, prompt, **kwargs):
        pass

    def request(self, prompt, **kwargs):
        return self.basic_request(prompt, **kwargs)

    def print_green(self, text: str, end: str = "\n"):
        return "\x1b[32m" + str(text) + "\x1b[0m" + end

    def print_red(self, text: str, end: str = "\n"):
        return "\x1b[31m" + str(text) + "\x1b[0m" + end

    def inspect_history(self, n: int = 1, skip: int = 0):
        provider: str = self.provider
        last_prompt = None
        printed = []
        n = n + skip
        for x in reversed(self.history[-100:]):
            prompt = x["prompt"]
            if prompt != last_prompt:
                if provider == "groq":
                    printed.append((prompt, x["response"]))
            last_prompt = prompt
            if len(printed) >= n:
                break
        printing_value = ""
        for idx, (prompt, choices) in enumerate(reversed(printed)):
            if (n - idx - 1) < skip:
                continue
            printing_value += "\n\n\n" + prompt
            text = ' ' + self._get_choice_text(choices).strip()
            printing_value += self.print_green(text, end="")
            if len(choices) > 1 and isinstance(choices, list):
                printing_value += self.print_red(f" \t (and {len(choices)-1} other completions)", end="")
            printing_value += "\n\n\n"
        print(printing_value)
        return printing_value

    @abstractmethod
    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        pass

    def copy(self, **kwargs):
        kwargs = {**self.kwargs, **kwargs}
        model = kwargs.pop("model")
        return self.__class__(model=model, **kwargs)


def backoff_hdlr(details):
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details),
    )


class GroqLM(LM):
    """Wrapper around groq's API."""

    def __init__(self, api_key: str, model: str = "mixtral-8x7b-32768", **kwargs):
        super().__init__(model)
        self.provider = "groq"
        if api_key:
            self.api_key = api_key
            self.client = Groq(api_key=api_key)
        else:
            raise ValueError("api_key is required for groq")

        self.kwargs = {
            "temperature": 0.0,
            "max_tokens": 150,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
            **kwargs,
        }
        models = self.client.models.list().data
        if models is not None:
            if model in [m.id for m in models]:
                self.kwargs["model"] = model
        self.history: list[dict[str, Any]] = []

    def log_usage(self, response):
        usage_data = response.usage
        if usage_data:
            total_tokens = usage_data.total_tokens
            logging.debug(f"Groq Total Tokens Response Usage: {total_tokens}")

    def basic_request(self, prompt: str, **kwargs):
        raw_kwargs = kwargs
        kwargs = {**self.kwargs, **kwargs}
        kwargs["messages"] = [{"role": "user", "content": prompt}]
        response = self.chat_request(**kwargs)
        history = {
            "prompt": prompt,
            "response": response.choices[0].message.content,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)
        return response

    @backoff.on_exception(
        backoff.expo,
        (groq.APIError, groq.RateLimitError),
        max_time=1000,
        on_backoff=backoff_hdlr,
    )
    def request(self, prompt: str, **kwargs):
        if "model_type" in kwargs:
            del kwargs["model_type"]
        return self.basic_request(prompt, **kwargs)

    def _get_choice_text(self, choice) -> str:
        return choice.message.content

    def chat_request(self, **kwargs):
        response = self.client.chat.completions.create(**kwargs)
        return response

    def __call__(self, prompt: str, only_completed: bool = True, return_sorted: bool = False, **kwargs):
        response = self.request(prompt, **kwargs)
        self.log_usage(response)
        choices = response.choices
        completions = [self._get_choice_text(c) for c in choices]
        if return_sorted and kwargs.get("n", 1) > 1:
            scored_completions = []
            for c in choices:
                tokens, logprobs = c["logprobs"]["tokens"], c["logprobs"]["token_logprobs"]
                if "" in tokens:
                    index = tokens.index("") + 1
                    tokens, logprobs = tokens[:index], logprobs[:index]
                avglog = sum(logprobs) / len(logprobs)
                scored_completions.append((avglog, self._get_choice_text(c)))
            scored_completions = sorted(scored_completions, reverse=True)
            completions = [c for _, c in scored_completions]
        return completions