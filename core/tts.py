from typing import AsyncGenerator

import numpy as np
from numpy.typing import NDArray

from config import settings
from .protocols import TTS


class OpenAITTS(TTS):
    """OpenAI Text-to-Speech

    Args:
        voice (str): The voice to use for TTS.
        model (str): The model to use for TTS.
        instructions (str): The instructions for the TTS.

    Raises:
        ImportError: If openai SDK is not installed.

    Example:
        ```
        tts = OpenAITTS()
        audio = await tts.tts("Hello, world!")

        audio_chunks = []
        async for chunk in tts.tts_stream("Hello, world!"):
            audio_chunks.append(chunk)

        audio = np.concatenate(audio_chunks)
        ```
    """

    def __init__(self, voice: str = "alloy", model: str = "gpt-4o-mini-tts"):
        """Initialize the TTS client."""
        self.voice = voice
        self.model = model
        self.instructions = "Speak in a cheerful and positive tone."
        self.client = self._init_client()

    def _init_client(self):
        try:
            # Try to import openai sdk
            from openai import AsyncClient
        except Exception:
            raise ImportError("openai is not installed, install it.")

        if not hasattr(settings, "openai_api_key"):
            raise ValueError(
                "API Key is not provided in settings, set the value with key: 'openai_api_key'"
            )

        # Create OpenAI AsyncClient
        api_key = settings.openai_api_key.get_secret_value()
        return AsyncClient(api_key=api_key)

    async def tts(self, text: str) -> NDArray[np.int16]:
        """Convert text to speech."""
        response = await self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
            instructions=self.instructions,
            response_format="pcm",
        )

        audio_int16 = np.frombuffer(response.content, dtype=np.int16)
        return audio_int16

    async def tts_stream(self, text: str) -> AsyncGenerator[NDArray[np.int16], None]:
        """Stream text to speech."""
        buffer = b""
        async with self.client.audio.speech.with_streaming_response.create(
            model=self.model,
            voice=self.voice,
            input=text,
            instructions=self.instructions,
            response_format="pcm",
        ) as response:
            async for chunk in response.iter_bytes():
                buffer += chunk
                # Process complete int16 samples (2 bytes each)
                while len(buffer) >= 2:
                    # Calculate how many complete samples we can process
                    complete_samples = len(buffer) // 2
                    bytes_to_process = complete_samples * 2

                    # Extract complete samples
                    complete_data = buffer[:bytes_to_process]
                    buffer = buffer[bytes_to_process:]

                    if complete_data:
                        yield np.frombuffer(complete_data, dtype=np.int16)

            # Handle any remaining data (shouldn't happen with proper PCM, but just in case)
            if buffer:
                # Pad with zero if odd number of bytes
                if len(buffer) % 2 == 1:
                    buffer += b"\x00"
                yield np.frombuffer(buffer, dtype=np.int16)
