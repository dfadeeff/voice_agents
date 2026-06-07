/**
 * Main app — connects mic capture, WebSocket, protobuf, and transcript UI.
 */

let ws = null;
let callActive = false;

const statusEl = document.getElementById("status");
const transcriptEl = document.getElementById("transcript");
const callBtn = document.getElementById("call-btn");
const btnIcon = document.getElementById("btn-icon");
const btnLabel = document.getElementById("btn-label");

// Accumulates streaming assistant text fragments
let currentAssistantEl = null;

function setStatus(state, text) {
  statusEl.className = "status " + state;
  statusEl.textContent = text;
}

function addMessage(role, text) {
  // Remove placeholder
  const ph = transcriptEl.querySelector(".transcript-placeholder");
  if (ph) ph.remove();

  const div = document.createElement("div");
  div.className = "msg " + role;

  const label = document.createElement("div");
  label.className = "label";
  label.textContent = role === "user" ? "Anrufer" : "Empfang";
  div.appendChild(label);

  const content = document.createElement("div");
  content.textContent = text;
  div.appendChild(content);

  transcriptEl.appendChild(div);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
  return content;
}

function appendAssistantText(text) {
  if (!currentAssistantEl) {
    currentAssistantEl = addMessage("assistant", text);
  } else {
    currentAssistantEl.textContent += text;
    transcriptEl.scrollTop = transcriptEl.scrollHeight;
  }
}

function finalizeAssistantMessage() {
  currentAssistantEl = null;
}

async function startCall() {
  callActive = true;
  callBtn.classList.add("active");
  btnIcon.textContent = "🔴";
  btnLabel.textContent = "Anruf beenden";
  setStatus("connecting", "Verbindung wird aufgebaut...");

  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${proto}//${location.host}/ws/call/new`;

  ws = new WebSocket(wsUrl);
  ws.binaryType = "arraybuffer";

  ws.onopen = async () => {
    setStatus("active", "Anruf aktiv");
    try {
      await AudioManager.startCapture((pcmBuffer) => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          const frame = PipecatProto.encodeAudioFrame(pcmBuffer, 16000, 1);
          ws.send(frame);
        }
      });
    } catch (err) {
      setStatus("error", "Mikrofonzugriff verweigert");
      endCall();
    }
  };

  ws.onmessage = (event) => {
    // JSON text messages (transcription + agent text from our custom processors)
    if (typeof event.data === "string") {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === "user_transcript" && msg.text) {
          finalizeAssistantMessage();
          addMessage("user", msg.text);
        } else if (msg.type === "agent_text" && msg.text) {
          appendAssistantText(msg.text);
        }
      } catch (e) { /* ignore non-JSON text */ }
      return;
    }

    // Binary protobuf messages (audio)
    if (!(event.data instanceof ArrayBuffer)) return;

    const frame = PipecatProto.decodeFrame(event.data);
    if (!frame) return;

    if (frame.type === "audio" && frame.audio) {
      AudioManager.playAudio(frame.audio, frame.sampleRate);
    }
  };

  ws.onclose = () => {
    if (callActive) {
      setStatus("idle", "Anruf beendet");
      resetUI();
    }
  };

  ws.onerror = () => {
    setStatus("error", "Verbindung fehlgeschlagen");
    resetUI();
  };
}

function endCall() {
  callActive = false;
  AudioManager.stopCapture();
  if (ws) {
    ws.close();
    ws = null;
  }
  setStatus("idle", "Anruf beendet");
  resetUI();
}

function resetUI() {
  callActive = false;
  callBtn.classList.remove("active");
  btnIcon.textContent = "📞";
  btnLabel.textContent = "Anruf starten";
  finalizeAssistantMessage();
}

function toggleCall() {
  if (callActive) {
    endCall();
  } else {
    startCall();
  }
}
