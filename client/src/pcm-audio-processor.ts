// PCM Audio Processor for handling raw PCM audio from the simplified server
// This replaces the Opus-based audio processor with direct PCM handling

declare var AudioWorkletProcessor: {
  prototype: typeof AudioWorkletProcessor;
  new (): typeof AudioWorkletProcessor;
};

declare var sampleRate: number;
declare var registerProcessor: (name: string, processorCtor: any) => void;

class PcmAudioProcessor extends AudioWorkletProcessor {
  private buffer: Float32Array[] = [];
  private initialBufferSamples: number;
  private maxDelay: number = 0;
  private minDelay: number = Infinity;
  private frameSize: number;

  constructor() {
    super();
    
    // Buffer length definitions - 80ms at 24kHz
    this.frameSize = 1920; // 24000 * 0.08
    this.initialBufferSamples = 2 * this.frameSize; // Wait for 2 frames before starting
    
    console.log("PCM Audio Processor initialized with frame size:", this.frameSize);
  }

  process(inputs: Float32Array[][], outputs: Float32Array[][], parameters: Record<string, Float32Array>): boolean {
    const output = outputs[0];
    const outputChannel = output[0];
    
    if (!outputChannel) {
      return true;
    }

    const currentSamples = this.currentSamples();
    const delay = currentSamples / sampleRate;
    
    if (this.canPlay()) {
      this.maxDelay = Math.max(this.maxDelay, delay);
      this.minDelay = Math.min(this.minDelay, delay);
    }

    // Fill output with zeros initially
    for (let i = 0; i < outputChannel.length; i++) {
      outputChannel[i] = 0;
    }

    // If we have enough buffered audio, play it
    if (this.buffer.length > 0) {
      const samplesToPlay = Math.min(outputChannel.length, this.buffer[0].length);
      
      for (let i = 0; i < samplesToPlay; i++) {
        outputChannel[i] = this.buffer[0][i];
      }
      
      // Remove played samples
      if (samplesToPlay >= this.buffer[0].length) {
        this.buffer.shift();
      } else {
        this.buffer[0] = this.buffer[0].slice(samplesToPlay);
      }
    }

    return true;
  }

  private currentSamples(): number {
    let totalSamples = 0;
    for (const chunk of this.buffer) {
      totalSamples += chunk.length;
    }
    return totalSamples;
  }

  private canPlay(): boolean {
    return this.currentSamples() >= this.initialBufferSamples;
  }

  // Method to add PCM data (called from main thread)
  addPcmData(pcmBytes: Uint8Array): void {
    // Convert bytes back to float32 samples
    const sampleCount = pcmBytes.length / 4;
    const samples = new Float32Array(sampleCount);
    const view = new DataView(pcmBytes.buffer, pcmBytes.byteOffset, pcmBytes.byteLength);
    
    for (let i = 0; i < sampleCount; i++) {
      samples[i] = view.getFloat32(i * 4, true); // true = little-endian
    }
    
    this.buffer.push(samples);
    
    // Log buffer status occasionally
    if (this.buffer.length % 10 === 0) {
      console.log(`PCM Buffer: ${this.buffer.length} chunks, ${this.currentSamples()} total samples`);
    }
  }

  // Method to get buffer statistics
  getBufferStats(): { chunks: number; samples: number; delay: number } {
    const samples = this.currentSamples();
    return {
      chunks: this.buffer.length,
      samples,
      delay: samples / sampleRate
    };
  }
}

// Register the processor
registerProcessor('pcm-audio-processor', PcmAudioProcessor); 