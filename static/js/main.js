const recordButton = document.getElementById("record-button");
const transcriptEl = document.getElementById("transcript");

let ws = null;
let micStream = null;
let audioContext, analyser, audioSource, animationFrame;
let micWorkletNode = null;
let ttsWorkletNode = null;
let userInitiatedStop = false;
let isRecording = false;
let isPlaying = false;
const visualizer = new AudioVisualizer();

// --- UI and State Management ---

function showError(message) {
    const toast = document.getElementById("error-toast");
    toast.textContent = message;
    toast.classList.add("show");
    setTimeout(() => {
        toast.classList.remove("show");
    }, 5000);
}

function updateButtonState(state) {
    switch (state) {
        case "recording":
            recordButton.classList.add("recording");
            recordButton.innerHTML = `<svg
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            color="#fff"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
            class="feather feather-mic"
          >
            <path
              d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"
            ></path>
            <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
            <line x1="12" y1="19" x2="12" y2="22"></line>
          </svg>`;
            isRecording = true;
            break;
        case "connecting":
            recordButton.classList.remove("recording");
            recordButton.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
    <linearGradient id="a11">
        <stop offset="0" stop-color="#000000ff" stop-opacity="0"></stop>
        <stop offset="1" stop-color="#020202ff"></stop>
    </linearGradient>
    <circle fill="none" stroke="url(#a11)" stroke-width="15" stroke-linecap="round" stroke-dasharray="0 44 0 44 0 44 0 44 0 360" cx="100" cy="100" r="70" transform-origin="center">
        <animateTransform type="rotate" attributeName="transform" calcMode="discrete" dur="2" values="360;324;288;252;216;180;144;108;72;36" repeatCount="indefinite"></animateTransform>
    </circle>
</svg>`;
            isRecording = false;
            break;
        case "idle":
        default:
            recordButton.classList.remove("recording");
            recordButton.innerHTML = `<svg
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            color="#000"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
            class="feather feather-mic"
          >
            <path
              d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"
            ></path>
            <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
            <line x1="12" y1="19" x2="12" y2="22"></line>
          </svg>`;
            isRecording = false;
            break;
    }
}


function setupAudioVisualization(stream) {
    visualizer.setup(stream);
}

// --- Core Logic: Recording and WebSocket Communication ---

async function initAudioContext() {
    if (!audioContext) {
        audioContext = new (window.AudioContext ||
            window.webkitAudioContext)();
    }

    // Resume AudioContext if suspended
    if (audioContext.state === "suspended") {
        await audioContext.resume();
    }
}

async function startRecording() {
    if (isRecording) return;
    userInitiatedStop = false;
    updateButtonState("connecting");

    try {
        // 1. Get user's audio stream
        micStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                noiseSuppression: true,
                echoCancellation: true,
            },
        });

        // 2. Connect to WebSocket
        const protocol =
            window.location.protocol === "https:" ? "wss://" : "ws://";
        const host = window.location.host;
        // NOTE: In a real app, the client_id should be dynamic
        const wsUrl = `${protocol}${host}/ws`;
        ws = new WebSocket(wsUrl);

        ws.onopen = async () => {
            console.log("WebSocket connection established.");

            try {
                // Initialize AudioContext first
                await initAudioContext();

                // Set up audio processing
                await setupAudioWorklet(micStream);
                setupAudioVisualization(micStream);

                // Init Audio Player
                await setupTTSPlayback();

                updateButtonState("recording");
            } catch (error) {
                console.error("Error setting up AudioWorklet:", error);
                showError(
                    `Failed to initialize audio processing: ${error.message}`
                );
                stopRecording();
            }
        };

        ws.onmessage = handleMessages;

        ws.onerror = (error) => {
            console.error("WebSocket Error:", error);
            showError("Connection error. Please try again.");
            stopRecording();
        };

        ws.onclose = (event) => {
            console.log(
                `WebSocket closed. Code: ${event.code}, Reason: ${event.reason}`
            );
            // Only show error if the closure was not initiated by the user.
            if (!userInitiatedStop) {
                showError("Connection lost. Please restart recording.");
            }
            stopRecording(); // Clean up everything
        };
    } catch (err) {
        console.error("Error getting user media:", err);
        showError(
            "Could not access microphone. Please grant permission and try again."
        );
        updateButtonState("idle");
    }
}

function stopRecording() {
    userInitiatedStop = true;

    // Disconnect and clean up AudioWorklet nodes
    if (micWorkletNode) {
        micWorkletNode.disconnect();
        micWorkletNode = null;
    }

    if (ttsWorkletNode) {
        ttsWorkletNode.disconnect();
        ttsWorkletNode = null;
    }

    // Stop all media tracks to release devices
    if (micStream) {
        micStream.getTracks().forEach((track) => track.stop());
        micStream = null;
    }

    if (ws) {
        // Let onclose handle the rest of the cleanup
        ws.close();
        ws = null;
    }
    if (animationFrame) {
        cancelAnimationFrame(animationFrame);
        animationFrame = null;
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
    updateButtonState("idle");
}

async function setupAudioWorklet(stream) {
    // Load AudioWorklet module
    await audioContext.audioWorklet.addModule(
        "/static/js/pcmWorkletProcessor.js"
    );
    // Create audio source from stream
    const source = audioContext.createMediaStreamSource(stream);

    // Create AudioWorkletNode
    micWorkletNode = new AudioWorkletNode(
        audioContext,
        "pcm-worklet-processor"
    );

    // Create identifier byte for source type
    const identifier = 1;

    // Buffer to accumulate audio data
    let audioBuffer = [];
    let bufferSize = 0;
    let headerBytes = 10;
    const targetBufferSize = 24000; // ~0.5 second at 48kHz (adjust as needed)

    // Handle messages from AudioWorklet
    micWorkletNode.port.onmessage = (event) => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            const pcmData = new Int16Array(event.data);

            // Add to buffer
            audioBuffer.push(pcmData);
            bufferSize += pcmData.length;

            // Send when buffer reaches target size
            if (bufferSize >= targetBufferSize && isPlaying === false) {
                // Combine all buffered chunks
                const combinedBuffer = new Int16Array(bufferSize);
                let offset = 0;

                for (const chunk of audioBuffer) {
                    combinedBuffer.set(chunk, offset);
                    offset += chunk.length;
                }

                // Create message with PCM data and identifier
                const dataBuffer = new ArrayBuffer(
                    combinedBuffer.byteLength + headerBytes
                );
                const dataView = new DataView(dataBuffer);
                const dataInt16 = new Int16Array(dataBuffer, headerBytes);

                // Create a timestamp bytes
                const ts = BigInt(Date.now());

                // Append identifier
                dataView.setUint16(0, identifier, false);
                // Append timestamp
                dataView.setBigUint64(2, ts, false);
                // Copy PCM data
                dataInt16.set(combinedBuffer);

                // Send the combined buffer
                ws.send(dataBuffer);

                // Reset buffer
                audioBuffer = [];
                bufferSize = 0;
            }
        }
    };

    // Connect the audio graph
    source.connect(micWorkletNode);
    micWorkletNode.connect(audioContext.destination);
    console.log(`AudioWorklet started.`);
}

async function setupTTSPlayback() {
    await audioContext.audioWorklet.addModule('/static/js/ttsPlaybackProcessor.js');
    ttsWorkletNode = new AudioWorkletNode(
        audioContext,
        'tts-playback-processor'
    );

    ttsWorkletNode.port.onmessage = (event) => {
        const { type } = event.data;
        if (type === 'ttsPlaybackStarted') {
            if (!isPlaying && ws && ws.readyState === WebSocket.OPEN) {
                isPlaying = true;
                console.log(
                    "TTS playback started. Reason: ttsWorkletNode Event ttsPlaybackStarted."
                );
                ws.send(JSON.stringify({ type: 'tts_start' }));
            }
        } else if (type === 'ttsPlaybackStopped') {
            if (isPlaying && ws && ws.readyState === WebSocket.OPEN) {
                isPlaying = false;
                console.log(
                    "TTS playback stopped. Reason: ttsWorkletNode Event ttsPlaybackStopped."
                );
                ws.send(JSON.stringify({ type: 'tts_stop' }));
            }
        }
    };
    ttsWorkletNode.connect(audioContext.destination);
    console.log(`TTSPlayback Started`);
}


async function handleMessages(event) {
    try {
        const message = JSON.parse(event.data);
        const { type, content, timestamp } = message;

        switch (type) {
            case "user.transcript.start":
                handleUserTranscriptStart();
                break;

            case "user.transcript.text.delta":
                handleUserTranscriptDelta(content, timestamp);
                break;

            case "user.transcript.end":
                handleUserTranscriptEnd();
                break;

            case "user.transcript.text":
                handleUserTranscript(content, timestamp);
                break;

            case "ai.response.text.start":
                handleAIResponseStart();
                break;

            case "ai.response.speech.start":
                handleAISpeechStart();
                break;

            case "ai.response.text.delta":
                handleAITextDelta(content, timestamp);
                break;

            case "ai.response.speech.delta":
                await handleAISpeechDelta(content, timestamp);
                break;

            case "ai.response.text.end":
                handleAIResponseEnd();
                break;

            case "ai.response.speech.end":
                handleAISpeechEnd();
                break;

            default:
                console.warn("Unknown message type:", type);
        }
    } catch (error) {
        console.error("Error handling message:", error);
    }
}

function handleUserTranscriptStart() {
    transcriptEl.innerText = "";
    console.log("User transcript started");
}

function handleUserTranscriptDelta(content, timestamp) {
    transcriptEl.innerText += content;
    if (transcriptEl.innerText.length > 100) {
        transcriptEl.innerText = "..." + transcriptEl.innerText.slice(-80); // Keep last 80 chars
    }
    console.log("User transcript delta:", content);
}

function handleUserTranscriptEnd() {
    console.log("User transcript ended");
}

function handleUserTranscript(content, timestamp) {
    console.log("User transcript:", content);
}

function handleAIResponseStart() {
    transcriptEl.innerText = "";
    console.log("AI response started");
}

function handleAISpeechStart() {
    console.log("AI speech started");
    // Stop visualizer during AI speech to avoid interference
    visualizer.isPlaying = false;
}

function handleAITextDelta(content, timestamp) {
    transcriptEl.innerText += content;
    if (transcriptEl.innerText.length > 100) {
        transcriptEl.innerText = "..." + transcriptEl.innerText.slice(-80); // Keep last 80 chars
    }
    console.log("AI text delta:", content);
}

async function handleAISpeechDelta(base64Audio, timestamp) {
    try {
        // Decode base64 audio
        const int16Data = base64ToInt16Array(base64Audio)

        // Resample the audio before sending it to the worklet
        const openAiSampleRate = 24000;
        const resampledData = await resampleInt16Array(int16Data, openAiSampleRate, audioContext.sampleRate);

        if (ttsWorkletNode) {
            ttsWorkletNode.port.postMessage(resampledData);
        }
    } catch (error) {
        console.error("Error handling speech delta:", error);
    }
}

function handleAIResponseEnd() {
    console.log("AI response ended");
}

function handleAISpeechEnd() {
    console.log("AI speech ended");
    // Resume visualizer after AI speech
    if (visualizer.analyser) {
        visualizer.isPlaying = true;
    }
}

function appendToTranscript(speaker, text, timestamp) { }

// In main.js, can be placed near the bottom with other utility functions

async function resampleInt16Array(int16Array, inputSampleRate, outputSampleRate) {
    if (inputSampleRate === outputSampleRate) {
        return int16Array;
    }

    // Convert Int16Array to Float32Array for the Web Audio API
    const float32Array = new Float32Array(int16Array.length);
    for (let i = 0; i < int16Array.length; i++) {
        float32Array[i] = int16Array[i] / 32768;
    }

    const offlineContext = new OfflineAudioContext(
        1, // Number of channels
        (float32Array.length * outputSampleRate) / inputSampleRate,
        outputSampleRate
    );

    const bufferSource = offlineContext.createBufferSource();
    const audioBuffer = offlineContext.createBuffer(1, float32Array.length, inputSampleRate);

    audioBuffer.copyToChannel(float32Array, 0);
    bufferSource.buffer = audioBuffer;
    bufferSource.connect(offlineContext.destination);
    bufferSource.start();

    const resampledBuffer = await offlineContext.startRendering();
    const resampledData = resampledBuffer.getChannelData(0);

    // Convert Float32Array back to Int16Array
    const resampledInt16Array = new Int16Array(resampledData.length);
    for (let i = 0; i < resampledData.length; i++) {
        resampledInt16Array[i] = Math.max(-1, Math.min(1, resampledData[i])) * 32767;
    }

    return resampledInt16Array;
}

// --- base64 to int16Data
function base64ToInt16Array(b64) {
    const raw = atob(b64);
    const buf = new ArrayBuffer(raw.length);
    const view = new Uint8Array(buf);
    for (let i = 0; i < raw.length; i++) {
        view[i] = raw.charCodeAt(i);
    }
    return new Int16Array(buf);
}

// --- Event Listener ---
recordButton.addEventListener("click", () => {
    if (!isRecording) {
        startRecording();
    } else {
        stopRecording();
    }
});

// Initial state
updateButtonState("idle");