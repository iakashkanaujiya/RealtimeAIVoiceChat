import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Pipeline specific config"""

    header_bytes: int = 10  # 2 Bytes + 8 Bytes for timestamp
    # VAD Config
    received_audio_sample_rate: int = 48000  # Incoming audio sample rate
    audio_sample_rate: int = 16000  # Model audio sample rate
    speech_pad_samples_ms: int = 100  # Pad each speech segment with 100ms of audio
    min_silence_duration_ms: int = (
        500  # Minimum duration of silence to consider the end of a speech segment
    )
    min_continuous_speech_s: int = (
        0.5  # Minimum duration of continuous speech to consider it valid
    )
    max_continuous_speech_s: int = (
        20  # Maximum duration of continuous speech to consider it valid
    )


@dataclass
class AudioChunk:
    flag: int
    timestamp: float
    audio: bytes | NDArray[np.int16]
    transcript: str | None = None
