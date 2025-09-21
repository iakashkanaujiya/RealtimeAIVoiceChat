# 🎤 Realtime AI Voice Chat

A sophisticated real-time voice conversation system featuring Emirates Airlines customer service AI agent. Built with FastAPI, WebSockets, and advanced AI models for seamless voice interactions.

## ✨ Features

### 🎯 Core Functionality

- **Real-time Voice Conversations**: Bi-directional voice communication with AI
- **Advanced Voice Activity Detection (VAD)**: Using Silero VAD for precise speech detection
- **Multiple AI Providers**: Support for OpenAI, Google Gemini, and Groq models
- **Streaming Responses**: Real-time text and audio streaming for natural conversations
- **Audio Visualization**: Beautiful real-time audio waveform visualization

### 🛫 Emirates Airlines Agent

- **Comprehensive Customer Service**: Flight bookings, status checks, seat changes
- **Smart Tools Integration**: Booking lookup, flight status, baggage assistance
- **Realistic Mock Data**: Pre-loaded with sample bookings and flight information
- **Multi-service Support**: Upgrades, meal preferences, compensation claims
- **Special Assistance**: Wheelchair, medical, dietary, and pet travel services

### 🎨 User Interface

- **Modern Web Interface**: Responsive design with space-themed animations
- **Audio Visualizer**: Real-time frequency analysis and waveform display
- **Error Handling**: User-friendly error messages and connection status
- **Mobile Responsive**: Works seamlessly on desktop and mobile devices

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- API keys for desired services (OpenAI, Google, Groq)

### Installation

#### Using UV (Recommended) ⚡

UV is a fast Python package installer and resolver, written in Rust. It's significantly faster than pip.

1. **Install UV**

```bash
# On Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

2. **Clone the repository**

```bash
git clone https://github.com/iakashkanaujiya/RealtimeAIVoiceChat.git
cd RealtimeAIVoiceChat
```

3. **Create virtual environment and install dependencies**

```bash
# Create venv and install all dependencies in one command
uv sync

# Or manually create venv and install
uv venv
uv sync
```

4. **Activate virtual environment**

```bash
# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

5. **Configure environment**

```bash
cp .env.example .env
# Edit .env with your API keys
```

6. **Run the application**

```bash
uv run main.py
# or
py main.py
```

### 🌐 Access the Application

Open your browser and navigate to `http://localhost:8000/chat` to start voice conversations!

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Development Settings
DEBUG=true
LOG_LEVEL=INFO
ALLOWED_ORIGINS=*

# Required: OpenAI API Key for TTS and STT
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Alternative providers
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### Audio Configuration

The system supports various audio configurations in `core/typing.py`:

```python
@dataclass
class PipelineConfig:
    received_audio_sample_rate: int = 48000  # Client audio sample rate
    audio_sample_rate: int = 16000           # Processing sample rate
    min_silence_duration_ms: int = 500       # Pause detection
    max_continuous_speech_s: int = 20        # Max speech duration
```

## 🏗️ Architecture

### Core Components

```
├── core/
│   ├── pipeline.py          # Main processing pipeline
│   ├── vad.py              # Voice Activity Detection
│   ├── stt.py              # Speech-to-Text providers
│   ├── tts.py              # Text-to-Speech providers
│   ├── agents/             # AI agents (Emirates customer service)
│   └── protocols.py        # Interface definitions
├── static/
│   ├── js/                 # Frontend JavaScript
│   └── css/               # Styling
├── templates/
│   └── index.html         # Web interface
└── main.py               # FastAPI server
```

### Data Flow

```
Audio Input → VAD → STT → AI Agent → TTS → Audio Output
     ↓          ↓     ↓       ↓        ↓         ↓
WebSocket → Queue → Queue → Queue → Queue → WebSocket
```

## 🎯 Usage Examples

### Basic Voice Interaction

1. Click the microphone button
2. Say: *"Hello, I need help with my flight"*
3. The AI agent will respond with voice and text
4. Continue the conversation naturally

### Emirates Agent Commands

```
"Check my booking EK123ABC"
"What's the status of flight EK215?"
"I want to change my seat to 15A"
"Cancel my flight booking"
"I need wheelchair assistance"
"What meals are available?"
```

### Programmatic Usage

```python
from core.pipeline import VoicePipeline
from core.agents import AirlineAgent
from core.stt import GroqSTT
from core.tts import OpenAITTS

# Create pipeline
pipeline = VoicePipeline(
    websocket=websocket,
    config=PipelineConfig(),
    stt=GroqSTT(),
    tts=OpenAITTS(),
    agent=AirlineAgent()
)

await pipeline.start()
```

## 🛠️ API Reference

### WebSocket Endpoints

#### `/ws` - Voice Communication

Real-time bidirectional voice communication endpoint.

**Message Types:**

- `user.transcript.start` - User starts speaking
- `user.transcript.text.delta` - Partial transcription
- `user.transcript.end` - User stops speaking
- `ai.response.text.start` - AI starts responding
- `ai.response.speech.delta` - Audio chunks
- `ai.response.text.end` - AI finishes responding

### REST Endpoints

#### `GET /` - API Status

Returns API welcome message and status.

#### `GET /chat` - Web Interface

Serves the voice chat web interface.

#### `GET /docs` - API Documentation

Interactive Swagger documentation.

## 🎨 Customization

### Adding New AI Agents

```python
from core.protocols import Agent

class CustomAgent(Agent):
    async def generate(self, message: str) -> str:
        # Your custom logic here
        return response
    
    async def generate_stream(self, message: str):
        # Streaming implementation
        yield chunk
```

### Custom TTS/STT Providers

```python
from core.protocols import TTS, STT

class CustomTTS(TTS):
    async def tts(self, text: str) -> NDArray[np.int16]:
        # Your TTS implementation
        return audio_array
```

### Audio Processing Pipeline

Modify `core/vad.py` for custom voice activity detection:

```python
class CustomVADProcessor(VADProcessor):
    async def process_audio_chunk(self, audio_chunk):
        # Custom processing logic
        return processed_chunk
```

## 🧪 Testing

### Mock Data

The Emirates agent includes realistic mock data:

- Sample bookings (EK123ABC, EK456DEF)
- Flight statuses (EK215, EK241, EK502, EK185)
- Comprehensive FAQ responses

### Test Conversations

```
User: "Check my booking EK123ABC"
Agent: "Booking found for Akash Kanaujiya: Flight EK215 - Dubai to London..."

User: "What's the status of flight EK241?"
Agent: "Flight EK241 is Delayed - Delayed by 30 minutes..."
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- **Silero Team** for excellent VAD models
- **OpenAI** for GPT and Whisper APIs
- **LangChain** for AI agent framework
- **FastAPI** for the web framework
- **Emirates Airlines** for inspiration (this is a demo project)
- **Astral** for the amazing UV package manager
