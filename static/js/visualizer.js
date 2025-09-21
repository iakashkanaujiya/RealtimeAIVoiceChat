class AudioVisualizer {
    constructor() {
        this.canvas = document.getElementById('visualizer');
        this.ctx = this.canvas.getContext('2d');
        this.audioContext = null;
        this.analyser = null;
        this.dataArray = null;
        this.source = null;
        this.isPlaying = false;

        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
        this.animate();
    }

    resizeCanvas() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * window.devicePixelRatio;
        this.canvas.height = rect.height * window.devicePixelRatio;
        this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        this.canvas.style.width = rect.width + 'px';
        this.canvas.style.height = rect.height + 'px';
    }

    setup(stream) {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        this.source = this.audioContext.createMediaStreamSource(stream);
        this.setupAnalyser();
        this.source.connect(this.analyser);
        this.isPlaying = true;
    }

    setupAnalyser() {
        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = 256;
        this.analyser.smoothingTimeConstant = 0.8;
        this.dataArray = new Uint8Array(this.analyser.frequencyBinCount);
    }

    stop() {
        if (this.source) {
            this.source.disconnect();
        }
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
        this.isPlaying = false;
    }

    drawWave(frequencies, offset = 0, opacity = 1, color = '#3b82f6') {
        const rect = this.canvas.getBoundingClientRect();
        const centerY = rect.height / 2;
        const width = rect.width;

        this.ctx.save();
        this.ctx.globalAlpha = opacity;
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 2;
        this.ctx.shadowBlur = 20;
        this.ctx.shadowColor = color;

        this.ctx.beginPath();

        for (let x = 0; x < width; x++) {
            const frequency = frequencies[Math.floor((x / width) * frequencies.length)] || 0;
            const amplitude = (frequency / 255) * (centerY * 0.8);
            const waveOffset = Math.sin((x / width) * Math.PI * 4 + offset) * 15;
            const y = centerY + Math.sin((x / width) * Math.PI * 3 + offset) * amplitude + waveOffset;

            if (x === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
        }

        this.ctx.stroke();
        this.ctx.restore();
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        const rect = this.canvas.getBoundingClientRect();
        this.ctx.fillStyle = "#030617";
        this.ctx.fillRect(0, 0, rect.width, rect.height);

        if (this.analyser && this.isPlaying) {
            this.analyser.getByteFrequencyData(this.dataArray);

            const time = Date.now() * 0.002;

            // Draw multiple flowing waves with different properties
            this.drawWave(this.dataArray, time * 2, 0.9, '#ffffff');
            this.drawWave(this.dataArray, time * 1.5 + 1, 0.7, '#60a5fa');
            this.drawWave(this.dataArray, time * 1.8 + 2, 0.5, '#3b82f6');
            this.drawWave(this.dataArray, time * 1.2 + 3, 0.3, '#1e40af');

            // Add inverted waves for symmetry
            this.drawInvertedWave(this.dataArray, time * 2.2, 0.6, '#60a5fa');
            this.drawInvertedWave(this.dataArray, time * 1.7 + 1, 0.4, '#3b82f6');
        } else {
            // Draw idle animation when no audio
            this.drawIdleAnimation();
        }
    }

    drawInvertedWave(frequencies, offset, opacity, color) {
        const rect = this.canvas.getBoundingClientRect();
        const centerY = rect.height / 2;
        const width = rect.width;

        this.ctx.save();
        this.ctx.globalAlpha = opacity;
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 1.5;
        this.ctx.shadowBlur = 15;
        this.ctx.shadowColor = color;

        this.ctx.beginPath();

        for (let x = 0; x < width; x++) {
            const frequency = frequencies[Math.floor((x / width) * frequencies.length)] || 0;
            const amplitude = (frequency / 255) * (centerY * 0.6);
            const waveOffset = Math.sin((x / width) * Math.PI * 5 + offset) * 10;
            const y = centerY - Math.sin((x / width) * Math.PI * 2.5 + offset) * amplitude - waveOffset;

            if (x === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
        }

        this.ctx.stroke();
        this.ctx.restore();
    }

    drawIdleAnimation() {
        const time = Date.now() * 0.001;
        const rect = this.canvas.getBoundingClientRect();
        const centerY = rect.height / 2;
        const width = rect.width;

        // Create multiple gentle flowing waves when idle
        const waves = [
            { color: '#3b82f6', opacity: 0.4, frequency: 2, amplitude: 15, speed: 1 },
            { color: '#60a5fa', opacity: 0.3, frequency: 1.5, amplitude: 10, speed: 0.8 },
            { color: '#1e40af', opacity: 0.2, frequency: 3, amplitude: 8, speed: 1.2 }
        ];

        waves.forEach(wave => {
            this.ctx.save();
            this.ctx.strokeStyle = wave.color;
            this.ctx.lineWidth = 2;
            this.ctx.globalAlpha = wave.opacity;
            this.ctx.shadowBlur = 10;
            this.ctx.shadowColor = wave.color;

            this.ctx.beginPath();
            for (let x = 0; x < width; x++) {
                const y = centerY + Math.sin((x / width) * Math.PI * wave.frequency + time * wave.speed) * wave.amplitude;
                if (x === 0) {
                    this.ctx.moveTo(x, y);
                } else {
                    this.ctx.lineTo(x, y);
                }
            }
            this.ctx.stroke();
            this.ctx.restore();
        });
    }
}
