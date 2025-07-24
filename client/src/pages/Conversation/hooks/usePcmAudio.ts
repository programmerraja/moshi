import { useCallback, useRef, useState } from "react";
import { useMediaContext } from "../MediaContext";

export enum PcmAudioStatuses {
  IDLE = "IDLE",
  READY = "READY",
  WAITING_FOR_PERMISSION = "WAITING_FOR_PERMISSION",
  ERROR = "ERROR",
  RECORDING = "RECORDING",
  STOPPED = "STOPPED",
  STOPPING = "STOPPING",
}

type usePcmAudioArgs = {
  constraints: MediaStreamConstraints;
  onDataChunk?: (chunk: Uint8Array) => void;
  onRecordingStart?: () => void;
  onRecordingStop?: () => void;
  sampleRate?: number;
  chunkSize?: number;
};

export const usePcmAudio = ({
  constraints,
  onDataChunk,
  onRecordingStart = () => {},
  onRecordingStop = () => {},
  sampleRate = 24000,
  chunkSize = 1920, // 80ms at 24kHz
}: usePcmAudioArgs) => {
  const { audioStreamDestination, audioContext, micDuration } = useMediaContext();
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<PcmAudioStatuses>(
    PcmAudioStatuses.IDLE,
  );

  const mediaStreamRef = useRef<MediaStream | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);

  const getMediaStream = useCallback(async () => {
    setStatus(PcmAudioStatuses.WAITING_FOR_PERMISSION);
    try {
      const stream =
        await window.navigator.mediaDevices.getUserMedia(constraints);
      setStatus(PcmAudioStatuses.IDLE);
      return stream;
    } catch (error: any) {
      console.error(error);
      setError(error.name);
      setStatus(PcmAudioStatuses.ERROR);
      return null;
    }
  }, [constraints, setStatus]);

  const startRecordingPcm = useCallback(async () => {
    console.log(Date.now() % 1000, "Starting PCM recording");
    const mediaStream = await getMediaStream();
    if (mediaStream) {
      mediaStreamRef.current = mediaStream;
      
      const analyser = audioContext.current.createAnalyser();
      const source = audioContext.current.createMediaStreamSource(mediaStream);
      source.connect(analyser);
      source.connect(audioStreamDestination.current);
      
      analyserRef.current = analyser;
      sourceRef.current = source;

      // Create script processor for raw PCM capture
      const processor = audioContext.current.createScriptProcessor(
        chunkSize, // buffer size
        1, // input channels
        1  // output channels
      );

      let chunk_idx = 0;
      let lastTime = Date.now();

      processor.onaudioprocess = (event) => {
        const inputBuffer = event.inputBuffer;
        const inputData = inputBuffer.getChannelData(0); // Get mono channel
        
        // Convert float32 to bytes (little-endian)
        const pcmBytes = new Uint8Array(inputData.length * 4);
        const view = new DataView(pcmBytes.buffer);
        
        for (let i = 0; i < inputData.length; i++) {
          view.setFloat32(i * 4, inputData[i], true); // true = little-endian
        }

        // Update duration
        const now = Date.now();
        micDuration.current += (now - lastTime) / 1000;
        lastTime = now;

        if (chunk_idx < 5) {
          console.log(Date.now() % 1000, "PCM Data chunk", chunk_idx++, inputData.length, micDuration.current);
        }

        if (onDataChunk) {
          onDataChunk(pcmBytes);
        }
      };

      processorRef.current = processor;
      source.connect(processor);
      processor.connect(audioContext.current.destination);

      setStatus(PcmAudioStatuses.RECORDING);
      onRecordingStart();

      return {
        analyser,
        mediaStream,
        source,
        processor,
      };
    }
    return {
      analyser: null,
      mediaStream: null,
      source: null,
      processor: null,
    };
  }, [setStatus, onDataChunk, onRecordingStart, chunkSize]);

  const stopRecording = useCallback(() => {
    setStatus(PcmAudioStatuses.STOPPING);
    
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    
    if (sourceRef.current) {
      sourceRef.current.disconnect();
      sourceRef.current = null;
    }
    
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }

    setStatus(PcmAudioStatuses.STOPPED);
    onRecordingStop();
  }, [setStatus, onRecordingStop]);

  return {
    status,
    error,
    startRecordingPcm,
    stopRecording,
  };
}; 