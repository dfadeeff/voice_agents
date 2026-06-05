/**
 * Mic capture (PCM 16-bit 16kHz mono) and audio playback.
 */

const AudioManager = (() => {
  const TARGET_SAMPLE_RATE = 16000;
  let audioContext = null;
  let mediaStream = null;
  let sourceNode = null;
  let workletNode = null;
  let playbackQueue = [];
  let isPlaying = false;

  async function startCapture(onAudioChunk) {
    audioContext = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });

    // Register the worklet processor inline via a blob
    const processorCode = `
      class PcmProcessor extends AudioWorkletProcessor {
        process(inputs) {
          const input = inputs[0];
          if (input.length > 0 && input[0].length > 0) {
            const float32 = input[0];
            const int16 = new Int16Array(float32.length);
            for (let i = 0; i < float32.length; i++) {
              const s = Math.max(-1, Math.min(1, float32[i]));
              int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
            }
            this.port.postMessage(int16.buffer, [int16.buffer]);
          }
          return true;
        }
      }
      registerProcessor("pcm-processor", PcmProcessor);
    `;
    const blob = new Blob([processorCode], { type: "application/javascript" });
    const url = URL.createObjectURL(blob);
    await audioContext.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);

    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: TARGET_SAMPLE_RATE,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      },
    });

    sourceNode = audioContext.createMediaStreamSource(mediaStream);
    workletNode = new AudioWorkletNode(audioContext, "pcm-processor");

    workletNode.port.onmessage = (event) => {
      onAudioChunk(event.data);
    };

    sourceNode.connect(workletNode);
    workletNode.connect(audioContext.destination);
  }

  function stopCapture() {
    if (workletNode) {
      workletNode.disconnect();
      workletNode = null;
    }
    if (sourceNode) {
      sourceNode.disconnect();
      sourceNode = null;
    }
    if (mediaStream) {
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
    }
    if (audioContext) {
      audioContext.close();
      audioContext = null;
    }
    playbackQueue = [];
    isPlaying = false;
  }

  function playAudio(pcmInt16Bytes, sampleRate) {
    if (!audioContext) return;
    playbackQueue.push({ pcmInt16Bytes, sampleRate });
    if (!isPlaying) drainQueue();
  }

  function drainQueue() {
    if (playbackQueue.length === 0 || !audioContext) {
      isPlaying = false;
      return;
    }
    isPlaying = true;
    const { pcmInt16Bytes, sampleRate } = playbackQueue.shift();

    const int16 = new Int16Array(
      pcmInt16Bytes.buffer,
      pcmInt16Bytes.byteOffset,
      pcmInt16Bytes.byteLength / 2
    );
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
      float32[i] = int16[i] / 32768;
    }

    const rate = sampleRate || TARGET_SAMPLE_RATE;
    const buffer = audioContext.createBuffer(1, float32.length, rate);
    buffer.getChannelData(0).set(float32);

    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContext.destination);
    source.onended = drainQueue;
    source.start();
  }

  function clearPlayback() {
    playbackQueue = [];
  }

  return { startCapture, stopCapture, playAudio, clearPlayback };
})();
