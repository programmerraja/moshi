import { useCallback, useRef, useState } from "react";
import { useMediaContext } from "../MediaContext";

export enum PcmServerAudioStatuses {
  IDLE = "IDLE",
  CONNECTING = "CONNECTING",
  CONNECTED = "CONNECTED",
  PLAYING = "PLAYING",
  ERROR = "ERROR",
  DISCONNECTED = "DISCONNECTED",
}

type usePcmServerAudioArgs = {
  onStatusChange?: (status: PcmServerAudioStatuses) => void;
  onError?: (error: string) => void;
};

export const usePcmServerAudio = ({
  onStatusChange = () => {},
  onError = () => {},
}: usePcmServerAudioArgs) => {
  const { audioContext } = useMediaContext();
  const [status, setStatus] = useState<PcmServerAudioStatuses>(
    PcmServerAudioStatuses.IDLE,
  );
  const [error, setError] = useState<string | null>(null);

  const audioNodeRef = useRef<AudioWorkletNode | null>(null);
  const processorRef = useRef<any>(null);

  const updateStatus = useCallback((newStatus: PcmServerAudioStatuses) => {
    setStatus(newStatus);
    onStatusChange(newStatus);
  }, [onStatusChange]);

  const handlePcmData = useCallback((pcmBytes: Uint8Array) => {
    if (processorRef.current && processorRef.current.port) {
      processorRef.current.port.postMessage({
        type: 'addPcmData',
        data: pcmBytes
      });
    }
  }, []);

  const initializeAudioProcessor = useCallback(async () => {
    try {
      // Load the PCM audio processor
      await audioContext.current.audioWorklet.addModule('/src/pcm-audio-processor.ts');
      
      // Create the audio worklet node
      const audioNode = new AudioWorkletNode(audioContext.current, 'pcm-audio-processor');
      audioNodeRef.current = audioNode;
      
      // Store reference to the processor for communication
      processorRef.current = audioNode;
      
      // Connect to audio output
      audioNode.connect(audioContext.current.destination);
      
      console.log("PCM Audio processor initialized successfully");
      return true;
    } catch (err) {
      console.error("Failed to initialize PCM audio processor:", err);
      setError("Failed to initialize audio processor");
      onError("Failed to initialize audio processor");
      return false;
    }
  }, [audioContext, onError]);

  const startAudio = useCallback(async () => {
    updateStatus(PcmServerAudioStatuses.CONNECTING);
    
    const success = await initializeAudioProcessor();
    if (success) {
      updateStatus(PcmServerAudioStatuses.CONNECTED);
    } else {
      updateStatus(PcmServerAudioStatuses.ERROR);
    }
  }, [updateStatus, initializeAudioProcessor]);

  const stopAudio = useCallback(() => {
    if (audioNodeRef.current) {
      audioNodeRef.current.disconnect();
      audioNodeRef.current = null;
    }
    processorRef.current = null;
    updateStatus(PcmServerAudioStatuses.DISCONNECTED);
  }, [updateStatus]);

  const getBufferStats = useCallback(() => {
    if (processorRef.current && processorRef.current.port) {
      processorRef.current.port.postMessage({ type: 'getBufferStats' });
    }
    return null;
  }, []);

  return {
    status,
    error,
    startAudio,
    stopAudio,
    handlePcmData,
    getBufferStats,
  };
}; 