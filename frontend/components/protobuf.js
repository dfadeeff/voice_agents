/**
 * Minimal protobuf encoder/decoder for Pipecat's Frame messages.
 *
 * Only handles the 3 frame types we need:
 *   - AudioRawFrame (field 2 in Frame oneof) — send mic audio, receive TTS audio
 *   - TranscriptionFrame (field 3 in Frame oneof) — receive transcriptions
 *   - TextFrame (field 1 in Frame oneof) — receive LLM text
 *
 * Wire format reference: https://protobuf.dev/programming-guides/encoding/
 */

const PipecatProto = (() => {
  // --- Low-level protobuf writing ---

  function encodeVarint(value) {
    const bytes = [];
    while (value > 0x7f) {
      bytes.push((value & 0x7f) | 0x80);
      value >>>= 7;
    }
    bytes.push(value & 0x7f);
    return bytes;
  }

  function encodeTag(fieldNumber, wireType) {
    return encodeVarint((fieldNumber << 3) | wireType);
  }

  function encodeLengthDelimited(fieldNumber, data) {
    const tag = encodeTag(fieldNumber, 2);
    const len = encodeVarint(data.byteLength);
    return [...tag, ...len, ...new Uint8Array(data)];
  }

  function encodeUint32(fieldNumber, value) {
    return [...encodeTag(fieldNumber, 0), ...encodeVarint(value)];
  }

  function encodeString(fieldNumber, str) {
    const encoded = new TextEncoder().encode(str);
    return encodeLengthDelimited(fieldNumber, encoded.buffer);
  }

  // --- Frame encoding ---

  function encodeAudioFrame(pcmBytes, sampleRate, numChannels) {
    // AudioRawFrame: audio=3(bytes), sample_rate=4(uint32), num_channels=5(uint32)
    const inner = [
      ...encodeLengthDelimited(3, pcmBytes),
      ...encodeUint32(4, sampleRate),
      ...encodeUint32(5, numChannels),
    ];
    const innerBuf = new Uint8Array(inner).buffer;
    // Frame: audio=2(AudioRawFrame)
    const outer = encodeLengthDelimited(2, innerBuf);
    return new Uint8Array(outer).buffer;
  }

  // --- Low-level protobuf reading ---

  function decodeVarint(buf, offset) {
    let result = 0;
    let shift = 0;
    let pos = offset;
    while (pos < buf.byteLength) {
      const byte = buf[pos];
      result |= (byte & 0x7f) << shift;
      pos++;
      if ((byte & 0x80) === 0) break;
      shift += 7;
    }
    return [result, pos];
  }

  function decodeFrame(arrayBuffer) {
    const buf = new Uint8Array(arrayBuffer);
    let pos = 0;
    let frameType = null;
    let frameData = null;

    while (pos < buf.byteLength) {
      const [tagValue, nextPos] = decodeVarint(buf, pos);
      pos = nextPos;
      const fieldNumber = tagValue >>> 3;
      const wireType = tagValue & 0x7;

      if (wireType === 2) {
        const [len, dataStart] = decodeVarint(buf, pos);
        pos = dataStart;
        const data = buf.slice(pos, pos + len);
        pos += len;

        // Frame oneof: text=1, audio=2, transcription=3, message=4, interruption=5
        if (fieldNumber >= 1 && fieldNumber <= 5) {
          frameType = fieldNumber;
          frameData = data;
        }
      } else if (wireType === 0) {
        const [, nextP] = decodeVarint(buf, pos);
        pos = nextP;
      }
    }

    if (!frameData) return null;

    if (frameType === 2) return decodeAudioRawFrame(frameData);
    if (frameType === 3) return decodeTranscriptionFrame(frameData);
    if (frameType === 1) return decodeTextFrame(frameData);
    return null;
  }

  function decodeAudioRawFrame(buf) {
    let pos = 0;
    let audio = null;
    let sampleRate = 0;
    let numChannels = 1;

    while (pos < buf.byteLength) {
      const [tagValue, nextPos] = decodeVarint(buf, pos);
      pos = nextPos;
      const fieldNumber = tagValue >>> 3;
      const wireType = tagValue & 0x7;

      if (wireType === 2) {
        const [len, dataStart] = decodeVarint(buf, pos);
        pos = dataStart;
        if (fieldNumber === 3) audio = buf.slice(pos, pos + len);
        pos += len;
      } else if (wireType === 0) {
        const [val, nextP] = decodeVarint(buf, pos);
        pos = nextP;
        if (fieldNumber === 4) sampleRate = val;
        if (fieldNumber === 5) numChannels = val;
      }
    }

    return { type: "audio", audio, sampleRate, numChannels };
  }

  function decodeTranscriptionFrame(buf) {
    let pos = 0;
    let text = "";
    let userId = "";

    while (pos < buf.byteLength) {
      const [tagValue, nextPos] = decodeVarint(buf, pos);
      pos = nextPos;
      const fieldNumber = tagValue >>> 3;
      const wireType = tagValue & 0x7;

      if (wireType === 2) {
        const [len, dataStart] = decodeVarint(buf, pos);
        pos = dataStart;
        const strBytes = buf.slice(pos, pos + len);
        const str = new TextDecoder().decode(strBytes);
        pos += len;
        if (fieldNumber === 3) text = str;
        if (fieldNumber === 4) userId = str;
      } else if (wireType === 0) {
        const [, nextP] = decodeVarint(buf, pos);
        pos = nextP;
      }
    }

    return { type: "transcription", text, userId };
  }

  function decodeTextFrame(buf) {
    let pos = 0;
    let text = "";

    while (pos < buf.byteLength) {
      const [tagValue, nextPos] = decodeVarint(buf, pos);
      pos = nextPos;
      const fieldNumber = tagValue >>> 3;
      const wireType = tagValue & 0x7;

      if (wireType === 2) {
        const [len, dataStart] = decodeVarint(buf, pos);
        pos = dataStart;
        const strBytes = buf.slice(pos, pos + len);
        pos += len;
        if (fieldNumber === 3) text = new TextDecoder().decode(strBytes);
        else pos; // skip other length-delimited fields
      } else if (wireType === 0) {
        const [, nextP] = decodeVarint(buf, pos);
        pos = nextP;
      }
    }

    return { type: "text", text };
  }

  return { encodeAudioFrame, decodeFrame };
})();
