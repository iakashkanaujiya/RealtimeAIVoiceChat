from typing import Protocol, AsyncGenerator
from numpy.typing import NDArray
import numpy as np


class STT(Protocol):
    """This is a protocol for Speech-to-Text."""

    # For non-streaming STT
    async def stt(self, audio: NDArray[np.int16], **kwargs) -> str: ...

    # For streaming STT
    async def stt_stream(
        self, audio: NDArray[np.int16], **kwargs
    ) -> AsyncGenerator[str, None]: ...


class TTS(Protocol):
    """This is a protocol for Text-to-Speech."""

    # For non-streaming TTS
    async def tts(self, text: str) -> NDArray[np.int16]: ...

    # For streaming TTS
    async def tts_stream(
        self, text: str
    ) -> AsyncGenerator[NDArray[np.int16], None]: ...


class Agent(Protocol):
    """This is a protocol for Agent."""

    # Message generation
    async def generate(self, message: str) -> str: ...

    # Streaming message generation
    async def generate_stream(self, message: str) -> AsyncGenerator[str, None]: ...
