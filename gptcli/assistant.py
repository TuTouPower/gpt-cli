import os
import sys
from attr import dataclass
import platform
from typing import Any, Dict, Iterator, Optional, TypedDict, List

from gptcli.completion import CompletionProvider, ModelOverrides, Message
from gptcli.google import GoogleCompletionProvider
from gptcli.llama import LLaMACompletionProvider
from gptcli.openai import OpenAICompletionProvider
from gptcli.anthropic import AnthropicCompletionProvider


class AssistantConfig(TypedDict, total=False):
    messages: List[Message]
    model: str
    temperature: float
    top_p: float


CONFIG_DEFAULTS = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "top_p": 1.0,
}

DEFAULT_ASSISTANTS: Dict[str, AssistantConfig] = {
    "dev": {
        "messages": [
            {
                "role": "system",
                "content": f"You are a helpful assistant who is an expert in software development. You are helping a user who is a software developer. Your responses are short and concise. You include code snippets when appropriate. Code snippets are formatted using Markdown with a correct language tag. User's `uname`: {platform.uname()}",
            },
            {
                "role": "user",
                "content": "Your responses must be short and concise. Do not include explanations unless asked.",
            },
            {
                "role": "assistant",
                "content": "Understood.",
            },
        ],
    },
    "flutter": {
        "messages": [
            {
                "role": "system",
                "content": "I want you to act as an expert Flutter and Dart engineer. You should ask for proficiency in the Dart language and experience with the Flutter SDK, along with understanding state management strategies such as Provider, Riverpod, or Bloc. Be sure to indicate the need for a comprehensive understanding of the Flutter rendering engine, widget tree, and layout design principles within Flutter.Request guidance in working with APIs, managing databases, and storage in a Flutter context. Include the need for expertise in conducting unit tests, widget tests, and integration tests.Request experience with continuous integration/continuous deployment (CI/CD) tools like Jenkins, GitHub Actions, or Travis CI. Additionally, ask for familiarity with handling app security and data privacy aspects to ensure the safety of your application's users.Seek assistance in building and releasing Flutter apps on both Android and iOS platforms, using tools like Visual Studio Code, Android Studio, and Xcode on Mac. Ask for knowledge about platform-specific design patterns, performance optimization techniques in Dart and Flutter, and effective use of Git for code versioning. Specify the need for experience with graphic design tools like Adobe XD or Sketch for UI/UX design. Also, indicate the need for familiarity with a specific methodology or workflow, like Agile, Scrum, or Kanban.Lastly, state that as part of a collaborative team environment, the engineer should have the ability to communicate effectively, understand project requirements clearly, and contribute to project documentation. Answers should be short, uncluttered, and must be written in Markdown unless I specifically request otherwise.",
            }
        ],
    },
    "python": {
        "messages": [
            {
                "role": "system",
                "content": "I want you to act as an expert Python engineer who has a deep understanding of large language models, backend development, and deep learning. You are proficient in Python programming, with a strong understanding of Python syntax, data structures, and libraries. You have a solid knowledge of machine learning and deep learning principles, including understanding of neural networks, backpropagation, and optimization algorithms. You are familiar with large language models such as GPT and BERT, including their architecture, training, and application. You have experience with backend development, including understanding of web frameworks like Flask or Django, database management, and API development. You can work with data, including data cleaning, manipulation, and analysis. You understand software development principles, including version control, testing, and debugging. You are proficient in Linux shell scripting and command-line tools. You understand computer networks, including protocols like HTTP and TCP/IP. You have knowledge of computer systems, including hardware and operating systems. You can answer questions such as how to build a backend system using Python and Flask, how GPT works and its practical applications, how to clean and prepare a dataset for machine learning, how backpropagation works in a neural network, how to debug a complex issue in a Python application, how to write a Linux shell script to automate a task, how the HTTP protocol works in a web application, how to manage version control in a collaborative software development project, and the role of TCP/IP in computer networking.",
            }
        ],
    },
    "general": {
        "messages": [],
    },
    "bash": {
        "messages": [
            {
                "role": "system",
                "content": f"You output only valid and correct shell commands according to the user's prompt. You don't provide any explanations or any other text that is not valid shell commands. User's `uname`: {platform.uname()}. User's `$SHELL`: {os.environ.get('SHELL')}.",
            }
        ],
    },
}


def get_completion_provider(model: str) -> CompletionProvider:
    if model.startswith("gpt"):
        return OpenAICompletionProvider()
    elif model.startswith("claude"):
        return AnthropicCompletionProvider()
    elif model.startswith("llama"):
        return LLaMACompletionProvider()
    elif model.startswith("chat-bison"):
        return GoogleCompletionProvider()
    else:
        raise ValueError(f"Unknown model: {model}")


class Assistant:
    def __init__(self, config: AssistantConfig):
        self.config = config

    @classmethod
    def from_config(cls, name: str, config: AssistantConfig):
        config = config.copy()
        if name in DEFAULT_ASSISTANTS:
            # Merge the config with the default config
            # If a key is in both, use the value from the config
            default_config = DEFAULT_ASSISTANTS[name]
            for key in [*config.keys(), *default_config.keys()]:
                if config.get(key) is None:
                    config[key] = default_config[key]

        return cls(config)

    def init_messages(self) -> List[Message]:
        return self.config.get("messages", [])[:]

    def supported_overrides(self) -> List[str]:
        return ["model", "temperature", "top_p"]

    def _param(self, param: str, override_params: ModelOverrides) -> Any:
        # If the param is in the override_params, use that value
        # Otherwise, use the value from the config
        # Otherwise, use the default value
        return override_params.get(
            param, self.config.get(param, CONFIG_DEFAULTS[param])
        )

    def complete_chat(
        self, messages, override_params: ModelOverrides = {}, stream: bool = True
    ) -> Iterator[str]:
        model = self._param("model", override_params)
        completion_provider = get_completion_provider(model)
        return completion_provider.complete(
            messages,
            {
                "model": model,
                "temperature": float(self._param("temperature", override_params)),
                "top_p": float(self._param("top_p", override_params)),
            },
            stream,
        )


@dataclass
class AssistantGlobalArgs:
    assistant_name: str
    model: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None


def init_assistant(
    args: AssistantGlobalArgs, custom_assistants: Dict[str, AssistantConfig]
) -> Assistant:
    name = args.assistant_name
    if name in custom_assistants:
        assistant = Assistant.from_config(name, custom_assistants[name])
    elif name in DEFAULT_ASSISTANTS:
        assistant = Assistant.from_config(name, DEFAULT_ASSISTANTS[name])
    else:
        print(f"Unknown assistant: {name}")
        sys.exit(1)

    # Override config with command line arguments
    if args.temperature is not None:
        assistant.config["temperature"] = args.temperature
    if args.model is not None:
        assistant.config["model"] = args.model
    if args.top_p is not None:
        assistant.config["top_p"] = args.top_p
    return assistant
