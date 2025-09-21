"""This module provides a basic chat functionality with LLM model."""

from typing import AsyncGenerator, List
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, AnyMessage

from .utils import init_chat_model
from .protocols import Agent
from ._prompts import CHAT_SYSTEM_PROMPT


class LLM(Agent):
    """This class implements a base LLM chat model.

    Args:
        model : Name of the model
        provider : Model provider, such as `openai`, `google_genai`, `groq`
        temperature : Model temperature
        system_prompt : System prompt for model
    """

    def __init__(
        self,
        model: str,
        provider: str,
        temperature: float,
        system_prompt: str,
    ):
        self.model = model
        self.temperature = temperature
        self.messages: List[AnyMessage] = [SystemMessage(system_prompt.strip())]
        self.llm = init_chat_model(self.model, provider, self.temperature)

    def _create_messages(self, human_message: str):
        self.messages.append(HumanMessage(human_message.strip()))
        self.messages = self.messages[-40:]
        return self.messages

    async def generate(self, message: str) -> str:
        """Generate LLM response"""
        messages = self._create_messages(message)
        response = await self.llm.ainvoke(messages)
        return response.content

    async def generate_stream(self, message: str) -> AsyncGenerator[str, None]:
        """Generate LLM response stream"""
        messages = self._create_messages(message)
        # Yield response chunk by chunk
        response_chunks = []
        async for chunk in self.llm.astream(messages):
            response_chunks.append(chunk.content)
            yield chunk.content

        self.messages.append(AIMessage("".join(response_chunks).strip()))


class OpenAILLM(LLM):
    """This class implements the openai chat model.

    Args:
        model : Name of the model, default is "gpt-5-mini"
        temperature : Model temperature
        system_prompt : System prompt for model
    """

    def __init__(
        self,
        model: str = "gpt-5-nano",
        temperature: float = 0.2,
        system_prompt: str = CHAT_SYSTEM_PROMPT,
    ):
        super().__init__(
            model,
            "openai",
            temperature,
            system_prompt,
        )


class GeminiLLM(LLM):
    """This class implements the googls's Gemini chat model.

    Args:
        model : Name of the model, default is "gemini-2.5-flash"
        temperature : Model temperature
        system_prompt : System prompt for model
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.2,
        system_prompt: str = CHAT_SYSTEM_PROMPT,
    ):
        super().__init__(
            model,
            "google_genai",
            temperature,
            system_prompt,
        )


class GroqLLM(LLM):
    """This class implements the Groq chat model.

    Args:
        model : Name of the model, default is `meta-llama/llama-4-scout-17b-16e-instruct`
        temperature : Model temperature
        system_prompt : System prompt for model
    """

    def __init__(
        self,
        model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
        temperature: float = 0.2,
        system_prompt: str = CHAT_SYSTEM_PROMPT,
    ):
        super().__init__(
            model,
            "groq",
            temperature,
            system_prompt,
        )
