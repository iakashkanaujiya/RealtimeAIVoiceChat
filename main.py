"""main.py
This module serves as the entry point for the Realtime AI Voice Chat server.
It initializes the FastAPI application and sets up the WebSocket route for real-time voice communication.
"""

from fastapi import FastAPI, WebSocket, WebSocketException, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from loguru import logger
from config import settings
from core.pipeline import VoicePipeline
from core.typing import PipelineConfig
from core.stt import GroqSTT
from core.tts import OpenAITTS
from core.agents import AirlineAgent


logger.level(settings.log_level)

# Initialize the FastAPI application
app = FastAPI(
    title="RealTimeAIVoiceChat",
    description="Real Time Voice Chat with AI",
    version="0.0.1",
    docs_url="/docs",
)
# Configure CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    """Root endpoint for the server."""
    return {"message": "Welcome to the Realtime AI Voice Chat Server!"}


@app.get("/chat")
async def demo():
    html_content = open("templates/index.html", encoding="utf-8").read()
    return HTMLResponse(html_content)


# --------------------------------------------------------
# WebSocket route for real-time voice communication
# --------------------------------------------------------
@app.websocket("/ws")
async def websocket_handler(websocket: WebSocket):
    try:
        await websocket.accept()
        logger.info("🖥️  Websocket connection accepted!")

        # Create processing pipeline
        pipeline = VoicePipeline(
            websocket=websocket,
            config=PipelineConfig(),
            stt=GroqSTT(),
            tts=OpenAITTS(),
            agent=AirlineAgent(model="openai/gpt-oss-120b", provider="groq"),
        )

        await pipeline.start()
    except WebSocketDisconnect:
        logger.info("Websocket disconnected")
    except WebSocketException as e:
        logger.exception(f"Websocket Exception {repr(e)}")
    except Exception as e:
        logger.exception(f"Exception {repr(e)}")
    finally:
        await pipeline.stop()
        await websocket.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.debug)
