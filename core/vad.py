"""
audio_in.py

This is core Audio Processing Module, and process the audio, detect speech
from audio and add processed audio into provided callback.
"""

import asyncio
from loguru import logger
from typing import List, Dict
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numpy.typing import NDArray
from torch import from_numpy, Tensor
from scipy.signal import resample_poly
from silero_vad import load_silero_vad, get_speech_timestamps

from .typing import AudioChunk


@lru_cache
def get_vad_model():
    """
    It loads the VAD model and caches it for reuse.
    """
    return load_silero_vad(onnx=True)


def run_vad_in_process(
    audio_tensor: Tensor,
    sample_rate: int,
    speech_pad_samples_ms: int,
) -> List[Dict[str, int]]:
    """
    This function runs voice activity detection in a separate process.

    Args:
        audio_tensor : Audio Tensor
        sample_rate : Audio Sample Rate, eg. 16000 for 16Hz
        speech_pad_samples_md : Speech pad smaples in ms
    """
    vad_model = get_vad_model()
    speech_timestamp = get_speech_timestamps(
        audio=audio_tensor,
        model=vad_model,
        sampling_rate=sample_rate,
        speech_pad_ms=speech_pad_samples_ms,
    )

    return speech_timestamp


def audio_resampler(audio_bytes: bytes, resample_ratio: int) -> NDArray[np.int16]:
    """
    Resample the audio, eg. 48Khz -> 16Khz

    Args:
        audio_bytes : Raw audio bytes
        resample_ratio : Audio resampling radios

    Returns:
        NDArray[np.int16] : Resampled ndarray of audip
    """
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)

    if np.max(np.abs(audio_int16)) == 0:
        # If all zeros, silent audio
        expect_len = int(np.ceil(len(audio_int16) / resample_ratio))
        return np.zeros(expect_len, np.int16)

    # Convert to float32 for resampling
    audio_float32 = audio_int16.astype(np.float32)
    # Resampling audio
    resampled_audio = resample_poly(audio_float32, 1, resample_ratio)
    # Converting back to int16, clipping to ensure data precision
    resampled_audio_int16 = np.clip(resampled_audio, -32768, 32767).astype(np.int16)

    return resampled_audio_int16


class VADProcessor:
    """
    VADProcessor: This is the implementation of the Voice Activity Detection class.
    This class implemeted Voice Activity Detection using silero vad which is enterprise grade
    voice acitivity detection model.

    Args:
        original_audio_sample_rate (int) : Original audio sample rate, e.g 48000 for 48Khz
        audio_sample_rate (int) : Audio sample rate for VAD model, e.g 16000 for 16Khz
        max_continuous_speech_s (int) : Maximum continuous speech duration in seconds
        min_continuous_speech_s (int) : Minimum continuous speech duration in seconds
        min_silence_duration_ms (int) : Minimum silence duration in milliseconds to consider end of speech
        speech_pad_samples_ms (int) : Pad each speech segment with this much of audio in milliseconds
    """

    def __init__(
        self,
        *,
        original_audio_sample_rate: int = 48000,
        audio_sample_rate: int = 16000,
        max_continuous_speech_s: int = 10,
        min_continuous_speech_s: int = 0.5,
        min_silence_duration_ms: int = 500,
        speech_pad_samples_ms: int = 100,
    ):
        self.original_audio_sample_rate = original_audio_sample_rate
        self.audio_sample_rate = audio_sample_rate
        self.max_continuous_speech_s = max_continuous_speech_s
        self.min_continuous_speech_s = min_continuous_speech_s
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_samples_ms = speech_pad_samples_ms

        # Audio buffer to hold audio samples
        self._audio_buffer = np.array([], dtype=np.int16)

        # Max size of the audio buffer
        self._max_audio_buffer = self.max_continuous_speech_s * self.audio_sample_rate

        # Min audio samples to detect VAD
        self._min_audio_samples = self.min_continuous_speech_s * self.audio_sample_rate

        # Min silence samples after speech detected
        self._min_silence_samples = self.min_silence_duration_ms * (
            self.audio_sample_rate // 1000
        )

        # Calculating resampling ratio
        self._resample_ratio = self.original_audio_sample_rate // self.audio_sample_rate

        # Getting event loop to run blocking operation
        self._event_loop = asyncio.get_running_loop()
        self._process_pool_executor = ProcessPoolExecutor(max_workers=2)

    def __del__(self):
        self._process_pool_executor.shutdown(wait=True)

    async def process_audio_chunk(self, audio_chunk: AudioChunk) -> AudioChunk | None:
        """
        Process the Audio Chunk and run the voice activity detection
        on audio samples, if speech is detected return the AudioChunk contain the
        speech samples, else return None.

        Args:
            audio_chunk : Audio Chunk
        """
        try:
            audio_bytes = audio_chunk.audio  # getting raw audio bytes
            audio_int16 = await self._resample_audio(audio_bytes)  # Resample audio

            # Combining audio samples
            self._audio_buffer = np.concatenate((self._audio_buffer, audio_int16))

            # If not enough audio to process
            if self._audio_buffer.size < self._min_audio_samples:
                return None

            # Detect speech
            speech_timestamps = await self._get_speech_timestamps()

            # If buffer is full we must process what we have regardless of the pause
            buffer_is_full = self._audio_buffer.size >= self._max_audio_buffer

            # If no speech is detected yet
            if not speech_timestamps and buffer_is_full:
                """
                When buffer is full and speech is not detected yet,
                then use the sliding window to trim the audio samples to 
                prevent memoery overflow.
                """
                samples_to_keep = int(self._audio_buffer.size * 0.9)
                self._audio_buffer = self._audio_buffer[-samples_to_keep:]
                return None

            if not speech_timestamps:
                return None

            """
            When speech is detected, find the minimum silence duration after end speech sample.
            If minimum silence samples are present after speech is detected, then it is considered 
            as pause detected.

            In case minimum silence duration is not present after the end speech sample, then it will be
            consider as pause regardless pause is detected or not.
            """
            speech_start_sample = speech_timestamps[0]["start"]
            speech_end_sample = speech_timestamps[-1]["end"]

            speech_samples = []  # Prepare speech samples
            for s_timestamp in speech_timestamps:
                _start, _end = s_timestamp["start"], s_timestamp["end"]
                speech_sample = self._audio_buffer[_start:_end]
                speech_samples.append(speech_sample)

            # Calculate the min silence samples after speech is detected
            silence_samples = self._audio_buffer.size - speech_end_sample

            if silence_samples >= self._min_silence_samples:
                self._audio_buffer = self._audio_buffer[speech_end_sample:]
                audio_chunk.audio = np.concatenate(speech_samples)
                return audio_chunk

            if buffer_is_full:
                # If buffer is full, then we must process what we have
                prev_length = self._audio_buffer.size
                self._audio_buffer = self._audio_buffer[speech_start_sample:]

                if prev_length == self._audio_buffer.size:
                    # If buffer is not reduced, then we must clear the buffer to prevent memory issue
                    self._audio_buffer = np.array([], dtype=np.int16)
                    audio_chunk.audio = np.concatenate(speech_samples)
                    return audio_chunk

            return None
        except Exception as e:
            logger.exception(f"Error in processing audio chunk: {repr(e)}")
            self._audio_buffer = np.array([], dtype=np.int16)
            raise

    async def _resample_audio(self, audio_bytes: bytes) -> NDArray[np.int16]:
        """
        Resample audio, e.g 48Khz to 16Khz, or 48Khz to 24Khz

        Agrs:
            audio_bytes (bytes) : Raw audio data received from client

        Returns:
            NdArray : NDArray of resampled audio data
        """
        resampled_audio = await self._event_loop.run_in_executor(
            self._process_pool_executor,
            audio_resampler,
            audio_bytes,
            self._resample_ratio,
        )

        return resampled_audio

    async def _get_speech_timestamps(self):
        """
        Get the speech timestamp form audio using Silero VAD model.

        Returns:
            List : List of detected timestamps
        """
        # Convert audio to float32
        audio_float32 = self._audio_buffer.astype(np.float32) / 32768.0
        # Create Tensor
        audio_tensor = from_numpy(audio_float32)

        # Running model to separate thread
        speech_timestamp = await self._event_loop.run_in_executor(
            self._process_pool_executor,
            run_vad_in_process,
            audio_tensor,
            self.audio_sample_rate,
            self.speech_pad_samples_ms,
        )

        return speech_timestamp
