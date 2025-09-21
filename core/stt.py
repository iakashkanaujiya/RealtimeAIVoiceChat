from typing import AsyncGenerator

import numpy as np
from numpy.typing import NDArray

from config import settings
from .utils import create_wav_buffer_from_bytes
from .protocols import STT


class OpenAISTT(STT):
    """
    This OpenAISTT model implements the STT model to convert the speech into text.
    This class uses the openai transcription endpoints to convert speech into text.

    Args:
        language (str): The language of the audio.
        model (str): The model to use for transcription.

    Raises:
        ImportError: If openai SDK is not installed

    Example:
        ```
        stt = OpenAISTT()
        text = await stt.stt(audio)

        audio_chunks = []
        async for chunk in stt.stt_stream(audio):
            audio_chunks.append(chunk)
        ```
    """

    def __init__(self, language: str = "en", model: str = "whisper-1"):
        self.language = language
        self.model = model
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

    async def stt(self, audio: NDArray[np.int16], **kwargs) -> str:
        """
        Args:
            audio (NDArray[np.int16]) : Numpy array of audio
            vad_config (VADConfig) : VAD Config

        Returns:
            str : Transcript
        """
        audio_bytes = audio.tobytes()
        config = kwargs.get("config")

        wav_buffer = create_wav_buffer_from_bytes(
            audio_bytes,
            config.audio_sample_rate,
        )

        transcription = await self.client.audio.transcriptions.create(
            file=wav_buffer,
            model=self.model,
            language=self.language,
            timeout=3.0,
        )

        return transcription.text

    async def stt_stream(
        self, audio: NDArray[np.int16], **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Args:
            audio (NDArray[np.int16]) : Numpy array of audio
            vad_config (VADConfig) : VAD Config

        Yields:
            str : Transcript chunk
        """

        if self.model == "whisper-1":
            # whisper-1 does not support stream
            transcription_text = await self.stt(audio, **kwargs)
            yield transcription_text

        audio_bytes = audio.tobytes()
        config = kwargs.get("config")

        wav_buffer = create_wav_buffer_from_bytes(
            audio_bytes,
            config.audio_sample_rate,
        )

        transcription = await self.client.audio.transcriptions.create(
            file=wav_buffer,
            model=self.model,
            language=self.language,
            timeout=3.0,
            stream=True,
        )

        async for chunk in transcription:
            if hasattr(chunk, "delta"):  # Check if chunk has delta attribute
                yield chunk.delta


class GroqSTT(STT):
    """
    This GroqSTT model implements the STT model to convert speech into text.
    This class uses the Groq transcription endpoints to convert text into speech.

    Args:
        language (str): The language of the audio.
        model (str): The model to use for transcription.

    Raises:
        ImportError: If groq SDK is not installed

    Example:
        ```
        stt = GroqSTT()
        text = await stt.stt(audio)

        audio_chunks = []
        async for chunk in stt.stt_stream(audio):
            audio_chunks.append(chunk)

        ```
    """

    def __init__(self, language: str = "en", model: str = "whisper-large-v3-turbo"):
        self.language = language
        self.model = model
        self.client = self._init_client()

    def _init_client(self):
        try:
            # Try to import openai sdk
            from groq import AsyncClient
        except Exception:
            raise ImportError("groq is not installed, install it.")

        if not hasattr(settings, "groq_api_key"):
            raise ValueError(
                "API Key is not provided in settings, set the value with key: 'groq_api_key'"
            )

        # Create Groq AsyncClient
        api_key = settings.groq_api_key.get_secret_value()
        return AsyncClient(api_key=api_key)

    async def stt(self, audio: NDArray[np.int16], **kwargs) -> str:
        """
        Args:
            audio (NDArray[np.int16]) : Numpy array of audio
            vad_config (VADConfig) : VAD Config

        Returns:
            str : Transcript
        """
        audio_bytes = audio.tobytes()
        config = kwargs.get("config")

        wav_buffer = create_wav_buffer_from_bytes(
            audio_bytes,
            config.audio_sample_rate,
        )

        transcription = await self.client.audio.transcriptions.create(
            file=wav_buffer,
            model=self.model,
            language=self.language,
            timeout=3.0,
        )

        return transcription.text

    async def stt_stream(
        self, audio: NDArray[np.int16], **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Args:
            audio (NDArray[np.int16]) : Numpy array of audio
            vad_config (VADConfig) : VAD Config

        Yields:
            str : Transcript chunk
        """

        # groq whisper model does not support streaming
        # so we will return the full transcription at once
        # to keep the interface consistent

        transcription_text = await self.stt(audio, **kwargs)
        yield transcription_text
