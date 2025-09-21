import json
import asyncio
import struct
import base64
from typing import Any, Dict
from datetime import datetime
from enum import Enum

from loguru import logger
from fastapi import WebSocket, WebSocketException

from .protocols import STT, TTS, Agent
from .typing import PipelineConfig, AudioChunk
from .vad import VADProcessor


class ResponseEventType(str, Enum):
    TRANSCRIPTION_START = "user.transcript.start"
    TRANSCRIPTION_TEXT = "user.transcript.text"  # Full text of the transcription
    TRANSCRIPTION_TEXT_DELTA = (
        "user.transcript.text.delta"  # Delta text of the transcription
    )
    TRANSCRIPTION_END = "user.transcript.end"
    AI_RESPONSE_TEXT_START = "ai.response.text.start"
    AI_RESPONSE_TEXT_DELTA = "ai.response.text.delta"
    AI_RESPONSE_TEXT_END = "ai.response.text.end"
    AI_RESPONSE_SPEECH_START = "ai.response.speech.start"
    AI_RESPONSE_SPEECH_DELTA = "ai.response.speech.delta"
    AI_RESPONSE_SPEECH_END = "ai.response.speech.end"


class VoicePipeline:
    """
    VoicePipeline manages the end-to-end processing of audio data
    from a WebSocket connection, including speech-to-text (STT),
    text-to-speech (TTS), and language model (LLM) interactions.

    Args:
        websocket (WebSocket): The websocket connection.
        config (PipelineConfig): The pipeline-specific configuration.
        stt (STT): The speech-to-text model.
        tts (TTS): The text-to-speech model.
        agent (Agent): The agent to manage interactions.
    """

    _MAX_QUEUE_SIZE = 60

    def __init__(
        self,
        *,
        websocket: WebSocket,
        config: PipelineConfig,
        stt: STT,
        tts: TTS,
        agent: Agent,
    ):
        self.websocket = websocket
        self.config = config
        self.stt = stt
        self.tts = tts
        self.agent = agent

        # Creating queue for holding data
        self.incoming_queue = asyncio.Queue(maxsize=self._MAX_QUEUE_SIZE)
        self.processed_audio_queue = asyncio.Queue(maxsize=self._MAX_QUEUE_SIZE)
        self.transcription_queue = asyncio.Queue(maxsize=self._MAX_QUEUE_SIZE)
        self.response_queue = asyncio.Queue(maxsize=self._MAX_QUEUE_SIZE)

        # Audio Processor that process Audio and detect speech
        self.vad_processor = VADProcessor(
            original_audio_sample_rate=self.config.received_audio_sample_rate,
            audio_sample_rate=self.config.audio_sample_rate,
            max_continuous_speech_s=self.config.max_continuous_speech_s,
            min_continuous_speech_s=self.config.min_continuous_speech_s,
            min_silence_duration_ms=self.config.min_silence_duration_ms,
            speech_pad_samples_ms=self.config.speech_pad_samples_ms,
        )

        # Tasks and state management
        self.tasks = []
        self.shutdown_event = asyncio.Event()

        # When tts is playing at client side
        self.is_tts_playing = False

    async def start(self):
        """Start the voice processing pipeline"""
        logger.info("🚀 Starting Voice Processing Pipeline")

        try:
            self.tasks = [
                asyncio.create_task(self.process_incoming_data(), name="incoming_data"),
                asyncio.create_task(
                    self.process_audio_chunk(), name="audio_processing"
                ),
                asyncio.create_task(
                    self.transcribe_audio_chunk(), name="transcription"
                ),
                asyncio.create_task(
                    self.generate_agent_response(), name="generate_agent_response"
                ),
                asyncio.create_task(
                    self.send_json_response(), name="send_json_response"
                ),
            ]

            # For testing: Start conversation with a prompt
            self._add_queue_no_wait(
                self.transcription_queue,
                AudioChunk(
                    flag=0,
                    timestamp=datetime.now().timestamp(),
                    audio=b"",
                    transcript="Start the conversation with Akash",
                ),
            )

            # Wait for first task to complete or shutdown signal
            done, _ = await asyncio.wait(
                self.tasks + [asyncio.create_task(self.shutdown_event.wait())],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Check if shutdown was requested
            if self.shutdown_event.is_set():
                logger.info("📡 Shutdown signal received")
            else:
                # Log which task completed first
                completed_task = next(iter(done))
                if hasattr(completed_task, "get_name"):
                    logger.warning(
                        f"⚠️ Task '{completed_task.get_name()}' completed unexpectedly"
                    )

        except Exception as e:
            logger.exception(f"🖥️ 💥 Error in Audio Processing Pipeline: {repr(e)}")
        finally:
            await self._cleanup()

    async def stop(self):
        """Gracefully stop the pipeline"""
        logger.info("🛑 Stopping pipeline...")
        self.shutdown_event.set()

    async def _cleanup(self):
        """Clean up resources and cancel tasks"""
        logger.info("🧹 Cleaning up pipeline resources...")
        self.is_running = False

        # Cancel all running tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
                logger.debug(f"🚫 Cancelled task: {getattr(task, '_name', 'unknown')}")

        # Wait for tasks to complete cancellation
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

        # Clear queues
        await self._clear_queues()

        logger.info("✅ Pipeline cleanup completed")

    async def _clear_queues(self):
        """Clear all queues to free memory"""
        queues = [
            self.incoming_queue,
            self.processed_audio_queue,
            self.transcription_queue,
        ]
        for queue in queues:
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

    def _add_queue_no_wait(self, queue: asyncio.Queue, item: Any):
        """Add item to queue without waiting. If the queue is full, log a warning."""
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            logger.warning("🚫 Queue is full, dropping audio chunk")

    def _create_event_message(
        self,
        type: str,
        content: str = None,
        timestamp: float = None,
    ) -> Dict[str, Any]:
        """
        Create a structured event message.

        Args:
            type (str): The type of the event.
            content (str, optional): The content of the event. Defaults to None.
            timestamp (float, optional): The timestamp of the event. If not provided, current time is used.
        """
        message = {
            "type": type,
            "timestamp": timestamp if timestamp else datetime.now().timestamp(),
        }

        if content:  # only add content if it's provided
            message["content"] = content

        return message

    def _add_event_message(
        self,
        type: str,
        content: str = None,
        timestamp: float = None,
    ):
        """
        Add an event message to the response queue.
        It will first create a structured message and then add to the queue.

        Args:
            type (str): The type of the event.
            content (str, optional): The content of the event. Defaults to None.
            timestamp (float, optional): The timestamp of the event. If not provided, current time is used.

        """
        message = self._create_event_message(type, content, timestamp)
        self._add_queue_no_wait(self.response_queue, message)

    # ================ Pipeline Tasks =================

    async def process_incoming_data(self):
        try:
            logger.debug("🖥️ ⚙️ process_incoming_data task started")
            while not self.shutdown_event.is_set():
                try:
                    # Add timeout to prevent indefinite blocking
                    data = await asyncio.wait_for(self.websocket.receive(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                if "bytes" in data and self.is_tts_playing is False:
                    raw_bytes = data["bytes"]

                    if len(raw_bytes) < self.config.header_bytes:
                        logger.debug("⚠️ Received invalid audio bytes")
                        continue

                    # 2Bytes for flag and 8Bytes for timestamp
                    flag, timestamp_ms = struct.unpack("!HQ", raw_bytes[:10])
                    pcm = raw_bytes[10:]

                    dt = datetime.fromtimestamp(timestamp_ms // 1_000)
                    ts = dt.timestamp()

                    # Create a audio chunk to hold the data
                    audio_chunk = AudioChunk(flag=flag, timestamp=ts, audio=pcm)
                    self._add_queue_no_wait(self.incoming_queue, audio_chunk)

                elif "text" in data:
                    data = json.loads(data["text"])

                    if data["type"] == "tts_start":
                        self.is_tts_playing = True
                    elif data["type"] == "tts_end":
                        self.is_tts_playing = False

        except asyncio.CancelledError:
            logger.debug("🚫 process_incoming_data task cancelled")
            raise
        except WebSocketException as e:
            logger.exception(
                f"💥 Websocket Exception at process_incoming_data: {repr(e)}"
            )
        except Exception as e:
            logger.exception(f"💥 Exception at process_incoming_data: {repr(e)}")

    async def process_audio_chunk(self):
        try:
            logger.debug("🖥️ 🔊 process_audio_chunk task started")
            while not self.shutdown_event.is_set():
                try:
                    audio_chunk: AudioChunk = await asyncio.wait_for(
                        self.incoming_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                processed_audio_chunk = await self.vad_processor.process_audio_chunk(
                    audio_chunk
                )

                if processed_audio_chunk:
                    self._add_queue_no_wait(
                        self.processed_audio_queue, processed_audio_chunk
                    )
        except asyncio.CancelledError:
            logger.debug("🚫 process_audio_chunk task cancelled")
            raise
        except Exception as e:
            logger.exception(f"💥 Exception at process_audio_chunk: {repr(e)}")

    async def transcribe_audio_chunk(self):
        try:
            logger.debug("🖥️ 💬 transcribe_audio_chunk task started")
            while not self.shutdown_event.is_set():
                try:
                    audio_chunk: AudioChunk = await asyncio.wait_for(
                        self.processed_audio_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                audio_int16 = audio_chunk.audio
                logger.debug(
                    f"🖥️ Transcribing ({len(audio_int16) * 2}) bytes of detected speech"
                )

                """
                Stream transcription results and send deltas to client
                """
                transcript = ""
                self._add_event_message(
                    ResponseEventType.TRANSCRIPTION_START,
                    timestamp=audio_chunk.timestamp,
                )
                async for chunk in self.stt.stt_stream(audio_int16, config=self.config):
                    self._add_event_message(
                        ResponseEventType.TRANSCRIPTION_TEXT_DELTA, chunk
                    )
                    transcript += chunk
                self._add_event_message(ResponseEventType.TRANSCRIPTION_END)

                # Clear audio to free memory and add transcript
                audio_chunk.audio = b""
                audio_chunk.transcript = transcript
                self._add_queue_no_wait(self.transcription_queue, audio_chunk)

        except asyncio.CancelledError:
            logger.debug("🚫 transcribe_audio_chunk task cancelled")
            raise
        except Exception as e:
            logger.exception(f"💥 Exception at transcribe_audio_chunk: {repr(e)}")

    async def generate_agent_response(self):
        """
        This task will run in the background and will consume
        data from the queue, and process it.
        """
        try:
            logger.debug("🖥️ 🤖 send_llm_response task started")
            while not self.shutdown_event.is_set():
                try:
                    audio_chunk: AudioChunk = await asyncio.wait_for(
                        self.transcription_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                transcript = audio_chunk.transcript
                logger.info(f"💬 Processing transcript: {transcript}")

                # Send full transcription text to client
                self._add_event_message(
                    ResponseEventType.TRANSCRIPTION_TEXT,
                    transcript,
                    audio_chunk.timestamp,
                )

                # Stream LLM response and feed to TTS in real-time
                text_buffer = ""
                sentence_endings = {".", "!", "?", "\n"}
                first_response_chunk = True
                first_speech_chunk = True

                async for chunk in self.agent.generate_stream(transcript):
                    text_buffer += chunk

                    if first_response_chunk:
                        # Send event when first chunk is generated
                        self._add_event_message(
                            ResponseEventType.AI_RESPONSE_TEXT_START
                        )
                        first_response_chunk = False

                    # Check if we have a complete sentence or phrase
                    if any(ending in text_buffer for ending in sentence_endings):
                        # Find the last sentence ending
                        last_ending_idx = max(
                            text_buffer.rfind(ending)
                            for ending in sentence_endings
                            if ending in text_buffer
                        )

                        # Extract complete sentence(s)
                        complete_text = text_buffer[: last_ending_idx + 1]
                        text_buffer = text_buffer[last_ending_idx + 1 :]

                        if complete_text:
                            # Send event when first speech chunk is generated
                            if first_speech_chunk:
                                self._add_event_message(
                                    ResponseEventType.AI_RESPONSE_SPEECH_START
                                )
                                first_speech_chunk = False

                            # Send text chunk to client
                            self._add_event_message(
                                ResponseEventType.AI_RESPONSE_TEXT_DELTA, complete_text
                            )
                            # Stream this text chunk to TTS
                            async for chunk in self.tts.tts_stream(complete_text):
                                array_bytes = chunk.tobytes()
                                base64_string = base64.b64encode(array_bytes).decode(
                                    "utf-8"
                                )

                                self._add_event_message(
                                    ResponseEventType.AI_RESPONSE_SPEECH_DELTA,
                                    base64_string,
                                )

                # Handle any remaining text
                if text_buffer.strip():
                    self._add_event_message(
                        ResponseEventType.AI_RESPONSE_TEXT_DELTA, text_buffer
                    )
                    async for chunk in self.tts.tts_stream(text_buffer):
                        array_bytes = chunk.tobytes()
                        base64_string = base64.b64encode(array_bytes).decode("utf-8")
                        self._add_event_message(
                            ResponseEventType.AI_RESPONSE_SPEECH_DELTA, base64_string
                        )

                self._add_event_message(ResponseEventType.AI_RESPONSE_TEXT_END)
                self._add_event_message(ResponseEventType.AI_RESPONSE_SPEECH_END)

        except asyncio.CancelledError:
            logger.info("🚫 send_llm_response task cancelled")
            raise
        except Exception as e:
            logger.exception(f"💥 Exception at send_llm_response: {repr(e)}")

    async def send_json_response(self):
        """
        This task will run in the background and will consume
        data (final response) from the queue, and send to the client.
        """
        try:
            logger.debug("🖥️ 🤖 send_json_response task started")
            while not self.shutdown_event.is_set():
                try:
                    response = await asyncio.wait_for(
                        self.response_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                await self.websocket.send_json(response)
        except asyncio.CancelledError:
            logger.info("🚫 send_json_response task cancelled")
            raise
        except Exception as e:
            logger.exception(f"💥 Exception at send_json_response: {repr(e)}")
