// static/pcmWorkletProcessor.js
class PCMWorkletProcessor extends AudioWorkletProcessor {
    process(inputs, outputs, parameters) {
        // Check if we have valid input
        if (inputs.length === 0 || !inputs[0] || inputs[0].length === 0) {
            return true;
        }

        const input = inputs[0];
        const inputChannel = input[0]; // Get first channel

        if (inputChannel && inputChannel.length > 0) {
            // convert Float32 → Int16 in the worklet
            const int16 = new Int16Array(inputChannel.length);
            for (let i = 0; i < inputChannel.length; i++) {
                let s = inputChannel[i];
                // Clamp the value to [-1, 1] range
                s = Math.max(-1, Math.min(1, s));
                // Convert to 16-bit PCM
                int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }

            // Send raw ArrayBuffer, transferable
            try {
                this.port.postMessage(int16.buffer, [int16.buffer]);
            } catch (error) {
                // If transfer fails, send a copy
                this.port.postMessage(int16.buffer.slice());
            }
        }
        return true;
    }
}

registerProcessor('pcm-worklet-processor', PCMWorkletProcessor);