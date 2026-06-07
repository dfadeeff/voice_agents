/**
 * Voice Agent UI — vanilla JS wired to Pipecat WebSocket.
 */

/* === Theme: dark 20:00–07:00, light otherwise === */
(function initTheme() {
  const hour = new Date().getHours();
  applyTheme(hour >= 20 || hour < 7 ? "dark" : "light");
})();

function applyTheme(t) {
  document.documentElement.classList.toggle("dark", t === "dark");
}

/* === DOM refs === */
const statusPill    = document.getElementById("status-pill");
const statusLabel   = document.getElementById("status-label");
const transcriptEl  = document.getElementById("transcript");
const callBtn       = document.getElementById("call-btn");
const btnLabel      = document.getElementById("btn-label");
const btnSvg        = document.getElementById("btn-icon-svg");
const viz           = document.getElementById("viz");
const toolbarTitle  = document.getElementById("toolbar-title");
const msgCountEl    = document.getElementById("msg-count");
const timerEl       = document.getElementById("timer");
const micBtn        = document.getElementById("mic-btn");

/* === State === */
let ws            = null;
let callActive    = false;
let callSeconds   = 0;
let callTimer     = null;
let messageCount  = 0;
let currentBubble = null;   // streaming assistant text target

/* === SVG paths === */
const PHONE_PATH = '<path d="M22 16.92v3a2 2 0 01-2.18 2 19.79 19.79 0 01-8.63-3.07 19.5 19.5 0 01-6-6 19.79 19.79 0 01-3.07-8.67A2 2 0 014.11 2h3a2 2 0 012 1.72c.127.96.361 1.903.7 2.81a2 2 0 01-.45 2.11L8.09 9.91a16 16 0 006 6l1.27-1.27a2 2 0 012.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0122 16.92z"/>';
const PHONE_OFF_PATH = '<line x1="1" y1="1" x2="23" y2="23"/><path d="M16.72 11.06A10.94 10.94 0 0119 12.55"/><path d="M5 4.34a19.79 19.79 0 00-2.88 7.83A2 2 0 014.11 2h3a2 2 0 012 1.72 12.84 12.84 0 00.7 2.81 2 2 0 01-.45 2.11L8.09 9.91"/><path d="M22 16.92v3a2 2 0 01-2.18 2 19.79 19.79 0 01-8.63-3.07 19.42 19.42 0 01-3.33-2.67"/>';

/* === Helpers === */
function pad2(n) { return String(n).padStart(2, "0"); }
function mmss(s)  { return pad2(Math.floor(s / 60)) + ":" + pad2(s % 60); }
function now()    { return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }); }

/* === Status === */
function setStatus(state, text) {
  statusPill.className = "status-pill " + state;
  statusLabel.textContent = text;

  const isLive = state === "active";
  viz.classList.toggle("active", isLive);
  micBtn.disabled = !isLive;

  if (state === "active")     toolbarTitle.textContent = "Anruf aktiv";
  else if (state === "connecting") toolbarTitle.textContent = "Verbindung…";
  else if (state === "error")      toolbarTitle.textContent = "Fehler";
  else toolbarTitle.textContent = messageCount > 0 ? "Anruf beendet" : "Bereit zum Telefonieren";
}

function updateMeta() {
  msgCountEl.textContent = messageCount;
  timerEl.textContent = callActive ? mmss(callSeconds) : "—";
}

/* === Timer === */
function startTimer() {
  callSeconds = 0;
  updateMeta();
  callTimer = setInterval(() => { callSeconds++; updateMeta(); statusLabel.textContent = "Aktiv · " + mmss(callSeconds); }, 1000);
}
function stopTimer() { if (callTimer) { clearInterval(callTimer); callTimer = null; } }

/* === Messages === */
function addBubble(role, text) {
  const row = document.createElement("div");
  row.className = "bubble-row " + (role === "user" ? "caller" : "agent");

  const wrap = document.createElement("div");
  wrap.className = "bubble-wrap";

  const label = document.createElement("div");
  label.className = "bubble-label";
  const roleName = role === "user" ? "Anrufer" : "Empfang";
  label.innerHTML = '<span>' + roleName + '</span><span style="color:var(--border)">&middot;</span><span class="time">' + now() + '</span>';

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  wrap.appendChild(label);
  wrap.appendChild(bubble);
  row.appendChild(wrap);
  transcriptEl.appendChild(row);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
  messageCount++;
  updateMeta();
  return bubble;
}

function appendAssistantText(text) {
  if (!currentBubble) {
    currentBubble = addBubble("assistant", text);
  } else {
    const current = currentBubble.textContent;
    const separator = current && !/\s$/.test(current) && !/^[,.;:!?]/.test(text) ? " " : "";
    currentBubble.textContent += separator + text;
    transcriptEl.scrollTop = transcriptEl.scrollHeight;
  }
}

function finalizeAssistant() { currentBubble = null; }

/* === Call lifecycle === */
async function startCall() {
  callActive = true;
  callBtn.classList.add("end");
  btnSvg.innerHTML = PHONE_OFF_PATH;
  btnLabel.textContent = "Anruf beenden";
  setStatus("connecting", "Verbindung…");

  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(proto + "//" + location.host + "/ws/call/new");
  ws.binaryType = "arraybuffer";

  ws.onopen = async () => {
    setStatus("active", "Aktiv · 00:00");
    startTimer();
    try {
      await AudioManager.startCapture((pcm) => {
        if (ws && ws.readyState === WebSocket.OPEN)
          ws.send(PipecatProto.encodeAudioFrame(pcm, 16000, 1));
      });
    } catch (_) {
      setStatus("error", "Mikrofon verweigert");
      endCall();
    }
  };

  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === "user_transcript" && msg.text) { finalizeAssistant(); addBubble("user", msg.text); }
        else if (msg.type === "agent_text" && msg.text)  { appendAssistantText(msg.text); }
      } catch (_) {}
      return;
    }
    if (!(ev.data instanceof ArrayBuffer)) return;
    const f = PipecatProto.decodeFrame(ev.data);
    if (f && f.type === "audio" && f.audio) AudioManager.playAudio(f.audio, f.sampleRate);
    if (f && f.type === "interruption") { AudioManager.interruptPlayback(); finalizeAssistant(); }
  };

  ws.onclose  = () => { if (callActive) { setStatus("idle", "Beendet"); resetUI(); } };
  ws.onerror  = () => { setStatus("error", "Fehlgeschlagen"); resetUI(); };
}

function endCall() {
  callActive = false;
  AudioManager.stopCapture();
  if (ws) { ws.close(); ws = null; }
  stopTimer();
  setStatus("idle", "Beendet");
  resetUI();
}

function resetUI() {
  callActive = false;
  stopTimer();
  callBtn.classList.remove("end");
  btnSvg.innerHTML = PHONE_PATH;
  btnLabel.textContent = "Anruf starten";
  finalizeAssistant();
}

function toggleCall() { callActive ? endCall() : startCall(); }
