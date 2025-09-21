import wave
import io

from langchain_core.language_models import BaseChatModel
from config import settings


MODEL_PROVIDERS = ["openai", "google_genai", "groq"]


def init_chat_model(
    model: str = "gpt-5-mini",
    model_provider: str = "openai",
    temperature: float = 0.2,
    api_key: str | None = None,
) -> BaseChatModel:
    """
    Initilize the chat model, if API key is not present raise excpetion.

    Args:
        model : Name of the model
        model_provider : Model provider, such as `openai`, `google_genai`, `groq`
        temperature : Model temperature from 0 to 1
        api_key : Model API key optional

    Raises:
        ValueError : If api key is not provided in settings or as parameter
        ValueError : If model provider is not supported
    """

    def load_api_key(key_name) -> str:
        if api_key:
            return api_key

        if not hasattr(settings, key_name):
            raise ValueError(
                f"API Key is not provided, either provide in settings as '{key_name}' or provide as parameter"
            )

        return getattr(settings, key_name).get_secret_value()

    """Initialize the chat model based on the provider."""

    if model_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=load_api_key("openai_api_key"),
            temperature=temperature,
        )

    if model_provider == "google_genai":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            api_key=load_api_key("google_api_key"),
            temperature=temperature,
        )

    if model_provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model,
            api_key=load_api_key("google_api_key"),
            temperature=temperature,
        )

    supported = ", ".join(MODEL_PROVIDERS)
    msg = (
        f"Unsupported {model_provider=}.\n\nSupported model providers are: {supported}"
    )
    raise ValueError(msg)


def create_wav_buffer_from_bytes(audio_bytes: bytes, sample_rate: int):
    """
    Creates a WAV buffer (bytes) from raw audio bytes.

    Args:
        audio_bytes (bytes): Raw audio data.
        sample_rate (int): Sample rate in Hz.

    Returns:
        bytes: The complete WAV file content as a bytes object.
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)

    buffer.seek(0)
    buffer.name = "audio.wav"
    return buffer
