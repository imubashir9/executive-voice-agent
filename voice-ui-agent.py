"""
Royal Cyber Executive Voice Agent
===================================
Combines:
  - Task 2: Deepgram dual-WebSocket STT + TTS (stutter-free, continuous)
  - Task 3: Azure OpenAI agent with Cal.com, Twilio SMS/WhatsApp, Email, CRM tools
"""

import os, json, httpx, asyncio, smtplib, threading, time, queue, re
from email.message import EmailMessage

import customtkinter as ctk
import websockets
import pyaudio
from openai import AzureOpenAI
from twilio.rest import Client as TwilioClient
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# Configuration
# ==========================================
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
AZURE_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_KEY        = os.getenv("AZURE_OPENAI_API_KEY", "")
DEPLOYMENT_NAME  = "gpt-4o-mini"

CAL_API_KEY      = os.getenv("CAL_API_KEY", "")
BASE_CAL_URL     = "https://api.cal.com/v2"

TWILIO_SID       = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN     = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE     = os.getenv("TWILIO_PHONE_NUMBER", "")
TWILIO_WHATSAPP  = os.getenv("TWILIO_WHATSAPP_NUMBER", "")
SMTP_EMAIL       = os.getenv("SMTP_EMAIL", "")
SMTP_PASSWORD    = os.getenv("SMTP_PASSWORD", "")

FORMAT   = pyaudio.paInt16
CHANNELS = 1
RATE     = 16000
CHUNK    = 2048

# ==========================================
# Azure OpenAI  (sync — wrapped in executor)
# ==========================================
llm_sync = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_KEY,
    api_version="2024-02-15-preview"
)

# ==========================================
# Cal.com v2 Helper
# ==========================================
def _cal_headers(version="2024-08-13"):
    """Standard headers required by every Cal.com v2 endpoint."""
    clean_key = CAL_API_KEY.replace("CAL_API_KEY", "").replace("=", "").strip()
    return {
        "Authorization":   f"Bearer {clean_key}",
        "Content-Type":    "application/json",
        "cal-api-version": version,
    }

async def make_cal_request(method, endpoint, data=None, params=None, version="2024-08-13"):
    url = f"{BASE_CAL_URL}{endpoint}"
    headers = _cal_headers(version)
    async with httpx.AsyncClient(timeout=30.0) as c:
        try:
            if method == "GET":
                r = await c.get(url, headers=headers, params=params)
            elif method == "POST":
                r = await c.post(url, headers=headers, json=data)
            elif method == "DELETE":
                r = await c.post(url, headers=headers, json=data)   # v2 cancel uses POST
            r.raise_for_status()
            return {"status": "success"} if r.status_code == 204 or not r.text else r.json()
        except httpx.HTTPStatusError as e:
            try:
                detail = e.response.json()
            except Exception:
                detail = e.response.text
            return {"error": f"{e.response.status_code}", "detail": detail}
        except Exception as e:
            return {"error": str(e)}

# ==========================================
# Tool Registry
# ==========================================
TOOLS = [
    {"type":"function","function":{"name":"list_event_types","description":"List all available event types and their IDs.","parameters":{"type":"object","properties":{},"required":[]}}},
    {"type":"function","function":{"name":"get_availability","description":"Check availability for a date range.","parameters":{"type":"object","properties":{"date_from":{"type":"string","description":"Start date YYYY-MM-DD"},"date_to":{"type":"string","description":"End date YYYY-MM-DD"}},"required":["date_from","date_to"]}}},
    {"type":"function","function":{"name":"book_meeting","description":"Book a meeting on the calendar.","parameters":{"type":"object","properties":{"event_type_id":{"type":"integer","description":"5223683 for 15-min, 5223682 for 30-min"},"start_time":{"type":"string","description":"ISO 8601 e.g. 2026-04-10T10:00:00+05:00"},"name":{"type":"string"},"email":{"type":"string"}},"required":["event_type_id","start_time","name","email"]}}},
    {"type":"function","function":{"name":"cancel_meeting","description":"Cancel an existing meeting by booking ID.","parameters":{"type":"object","properties":{"booking_id":{"type":"integer"},"reason":{"type":"string"}},"required":["booking_id"]}}},
    {"type":"function","function":{"name":"send_sms","description":"Send an SMS via Twilio.","parameters":{"type":"object","properties":{"to_number":{"type":"string"},"message":{"type":"string"}},"required":["to_number","message"]}}},
    {"type":"function","function":{"name":"send_whatsapp","description":"Send a WhatsApp message via Twilio.","parameters":{"type":"object","properties":{"to_number":{"type":"string"},"message":{"type":"string"}},"required":["to_number","message"]}}},
    {"type":"function","function":{"name":"send_email","description":"Send an email via SMTP.","parameters":{"type":"object","properties":{"to_email":{"type":"string"},"subject":{"type":"string"},"body":{"type":"string"}},"required":["to_email","subject","body"]}}},
    {"type":"function","function":{"name":"create_lead","description":"Save a new CRM lead.","parameters":{"type":"object","properties":{"name":{"type":"string"},"email":{"type":"string"},"phone":{"type":"string"},"notes":{"type":"string"}},"required":["name","email"]}}},
    {"type":"function","function":{"name":"log_call_summary","description":"Log a call summary.","parameters":{"type":"object","properties":{"caller_name":{"type":"string"},"summary":{"type":"string"}},"required":["caller_name","summary"]}}},
]

SYSTEM_PROMPT = """ROLE & IDENTITY: You are a friendly, highly capable Executive AI Assistant for Mubashir. Your job is to help users manage their calendar, send communications (SMS, WhatsApp, Email), and manage CRM data.

GENERAL RULES:
- Keep responses concise, conversational, and natural. Speak in short sentences — your reply will be read aloud.
- When a tool returns data (event types, slots, bookings), ALWAYS summarize it to the user in plain spoken language. Never say you could not retrieve it if the tool returned a result.
- When list_event_types returns data, read the "title" and "length" fields and describe each event type by name and duration.
- Never read raw JSON to the user.
- Always convert 24-hour time to spoken 12-hour time (e.g. 14:00 → 2 PM).
- Only confirm an action if the tool returned a success response.

HOW TO USE YOUR TOOLS:
1. Scheduling: list_event_types → get_availability → book_meeting / cancel_meeting. Use ISO 8601 Asia/Karachi timezone (e.g. 2026-04-07T10:00:00+05:00). For cancel_meeting, always use the booking UID string (e.g. 'abc123xyz') that was returned when the meeting was created — never a numeric ID.
2. Messaging: send_sms, send_whatsapp, send_email. Confirm once sent.
3. CRM: create_lead, log_call_summary.

CRITICAL: If availability check fails but user gave a clear time, skip it and call book_meeting directly with event_type_id 5223683."""

# ==========================================
# Tool Executor
# ==========================================
async def execute_tool(name, args):
    print(f"[TOOL] ⚙️  {name}({args})")
    try:
        if name == "list_event_types":
            result = await make_cal_request("GET", "/event-types")
            print(f"[TOOL RESULT] {str(result)[:400]}")
            return json.dumps(result)

        elif name == "get_availability":
            # v2 slots: use eventTypeId + start/end date strings + correct version header
            params = {
                "eventTypeId": 5223683,
                "start":       args["date_from"],   # YYYY-MM-DD, Deepgram resolves tz
                "end":         args["date_to"],
                "timeZone":    "Asia/Karachi",
            }
            result = await make_cal_request("GET", "/slots", params=params, version="2024-09-04")
            print(f"[TOOL RESULT] {str(result)[:400]}")
            return json.dumps(result)

        elif name == "book_meeting":
            # v2 bookings: attendee nested object, start in UTC ISO 8601
            from datetime import datetime, timezone, timedelta
            # Convert Asia/Karachi (+05:00) ISO string to UTC if needed
            start_str = args["start_time"]
            # Accept both +05:00 and Z formats; normalise to UTC Z string
            try:
                if "+" in start_str or (start_str.endswith("Z") is False and "-" in start_str[10:]):
                    # naive parse then shift to UTC
                    from dateutil import parser as dtparser
                    dt = dtparser.parse(start_str)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone(timedelta(hours=5)))
                    utc_start = dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                else:
                    utc_start = start_str
            except Exception:
                utc_start = start_str  # fall back to whatever the model gave

            payload = {
                "eventTypeId": args["event_type_id"],
                "start": utc_start,
                "attendee": {
                    "name":     args["name"],
                    "email":    args["email"],
                    "timeZone": "Asia/Karachi",
                    "language": "en",
                },
                "metadata": {},
            }
            result = await make_cal_request("POST", "/bookings", data=payload)
            print(f"[TOOL RESULT] {str(result)[:400]}")
            return json.dumps(result)

        elif name == "cancel_meeting":
            # v2 cancel: POST to /bookings/{uid}/cancel  (uid is a string like abc123xyz)
            # The model may pass either a numeric ID or a UID string — handle both
            uid = str(args["booking_id"])
            result = await make_cal_request(
                "DELETE",                        # mapped to POST inside make_cal_request
                f"/bookings/{uid}/cancel",
                data={"cancellationReason": args.get("reason", "Cancelled by Agent")},
            )
            return json.dumps(result)

        elif name == "send_sms":
            if not TWILIO_SID:
                return json.dumps({"status": "simulated_success"})
            c = TwilioClient(TWILIO_SID, TWILIO_TOKEN)
            m = c.messages.create(body=args["message"], from_=TWILIO_PHONE, to=args["to_number"])
            return json.dumps({"status": "success", "sid": m.sid})

        elif name == "send_whatsapp":
            if not TWILIO_SID:
                return json.dumps({"status": "simulated_success"})
            c = TwilioClient(TWILIO_SID, TWILIO_TOKEN)
            m = c.messages.create(body=args["message"],
                from_=f"whatsapp:{TWILIO_WHATSAPP}", to=f"whatsapp:{args['to_number']}")
            return json.dumps({"status": "success", "sid": m.sid})

        elif name == "send_email":
            if not SMTP_PASSWORD:
                return json.dumps({"status": "simulated_success"})
            msg = EmailMessage()
            msg.set_content(args["body"])
            msg["Subject"] = args["subject"]
            msg["From"]    = SMTP_EMAIL
            msg["To"]      = args["to_email"]
            with smtplib.SMTP("smtp.gmail.com", 587) as s:
                s.starttls(); s.login(SMTP_EMAIL, SMTP_PASSWORD); s.send_message(msg)
            return json.dumps({"status": "success"})

        elif name == "create_lead":
            with open("leads.json", "a") as f:
                f.write(json.dumps(args) + "\n")
            return json.dumps({"status": "success"})

        elif name == "log_call_summary":
            with open("call_logs.txt", "a") as f:
                f.write(f"Caller: {args['caller_name']} | Summary: {args['summary']}\n")
            return json.dumps({"status": "success"})

        return json.dumps({"error": f"Unknown tool: {name}"})
    except Exception as e:
        return json.dumps({"error": str(e)})

# ==========================================
# Agent Orchestrator
# ==========================================
async def run_agent_async(user_text, history):
    messages = history[:]
    if not messages:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_text})

    loop = asyncio.get_event_loop()
    while True:
        response = await loop.run_in_executor(
            None,
            lambda: llm_sync.chat.completions.create(
                model=DEPLOYMENT_NAME, messages=messages,
                tools=TOOLS, tool_choice="auto"
            )
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                result = await execute_tool(tc.function.name, json.loads(tc.function.arguments))
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                  "name": tc.function.name, "content": result})
            continue   # give model a chance to react

        reply = msg.content or ""
        messages.append({"role": "assistant", "content": reply})
        return reply, messages

# ==========================================
# Stutter-Free Audio Player
# ==========================================
class ContinuousAudioPlayer:
    def __init__(self, unlock_callback):
        self.q = queue.Queue()
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)
        self.unlock_callback = unlock_callback
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            item = self.q.get()
            if item == b"UNLOCK":
                self.unlock_callback()
            else:
                self.stream.write(item)

    def play(self, data):   self.q.put(data)
    def unlock(self):       self.q.put(b"UNLOCK")

# ==========================================
# Main Application
# ==========================================
class VoiceAgentApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("🤖 Royal Cyber Executive Voice Agent")
        self.geometry("620x780")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # ── Header labels ──────────────────────────────────────────────
        self.status_lbl = ctk.CTkLabel(self, text="Status: Offline",
            font=("Arial", 18, "bold"), text_color="gray")
        self.status_lbl.pack(pady=(15, 0))

        self.latency_lbl = ctk.CTkLabel(self, text="Response Time: -- ms",
            font=("Arial", 14), text_color="cyan")
        self.latency_lbl.pack(pady=(4, 8))

        # ── Chat box ───────────────────────────────────────────────────
        self.chat = ctk.CTkTextbox(self, width=570, height=560,
            font=("Arial", 13), wrap="word")
        self.chat.pack(pady=6, padx=15)

        # Colour tags for You / Agent bubbles
        self.chat._textbox.tag_configure("you_tag",
            foreground="#00ff99", spacing1=10, spacing3=6)
        self.chat._textbox.tag_configure("agent_tag",
            foreground="#00bfff", spacing1=10, spacing3=6)
        self.chat._textbox.tag_configure("separator",
            foreground="#444455", spacing1=2, spacing3=2)

        self._insert_chat("System ready. Press 'Start Listening' to begin.\n", "separator")
        self.chat.configure(state="disabled")

        # ── Buttons ────────────────────────────────────────────────────
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=10)

        self.start_btn = ctk.CTkButton(btn_frame, text="🎙️ Start Listening",
            command=self.start, fg_color="green", hover_color="darkgreen")
        self.start_btn.grid(row=0, column=0, padx=10)

        self.stop_btn = ctk.CTkButton(btn_frame, text="⏹ Stop",
            command=self.stop, fg_color="red", hover_color="darkred", state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=10)

        # ── State ──────────────────────────────────────────────────────
        self.is_running     = False
        self.agent_busy     = False
        self.t0             = None
        self.is_first_audio = False
        self.conv_history   = []
        self._mic_open_at   = 0.0   # timestamp when mic last re-opened after agent spoke

        self.loop       = None
        self.tts_queue  = None
        self.audio_player = ContinuousAudioPlayer(unlock_callback=self._on_audio_done)

    # ── Chat helpers ───────────────────────────────────────────────────
    def _insert_chat(self, text, tag):
        """Low-level insert; always call from main thread."""
        self.chat.configure(state="normal")
        self.chat._textbox.insert("end", text, tag)
        self.chat._textbox.see("end")
        self.chat.configure(state="disabled")

    def _append_you(self, text):
        def _do():
            self._insert_chat(f"🧑 You\n{text}\n", "you_tag")
            self._insert_chat("─" * 60 + "\n", "separator")
        self.after(0, _do)

    def _append_agent(self, text):
        def _do():
            self._insert_chat(f"🤖 Agent\n{text}\n", "agent_tag")
            self._insert_chat("─" * 60 + "\n", "separator")
        self.after(0, _do)

    # ── Status / latency helpers ───────────────────────────────────────
    def _set_status(self, text, color):
        self.after(0, lambda: self.status_lbl.configure(text=text, text_color=color))

    def _set_latency(self, ms):
        self.after(0, lambda: self.latency_lbl.configure(text=f"Response Time: {ms:.0f} ms"))

    # ── Audio unlock callback ──────────────────────────────────────────
    def _on_audio_done(self):
        # Wait 600 ms after speaker stops before re-enabling mic.
        # This drains any room echo / reverb so the STT doesn't
        # pick up the tail end of the agent's own voice.
        def _delayed_unlock():
            time.sleep(0.6)
            self._mic_open_at = time.time()
            self.agent_busy = False
            if self.is_running:
                self._set_status("🎙️ Listening...", "#00ff00")
        threading.Thread(target=_delayed_unlock, daemon=True).start()

    # ── Start / Stop ───────────────────────────────────────────────────
    def start(self):
        self.is_running = True
        self.agent_busy = False
        self.conv_history = []
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self._set_status("🎙️ Listening...", "#00ff00")
        threading.Thread(target=self._run_loop, daemon=True).start()

    def stop(self):
        self.is_running = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self._set_status("Status: Offline", "gray")

    def _run_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.tts_queue = asyncio.Queue()
        try:
            self.loop.run_until_complete(asyncio.gather(
                self._stt_websocket(),
                self._tts_websocket(),
            ))
        except Exception as e:
            print(f"[LOOP ERROR] {e}")

    # ── WebSocket 1 — Deepgram STT ─────────────────────────────────────
    async def _stt_websocket(self):
        """
        Keeps a PERMANENT Deepgram STT connection.
        - Always streams real mic audio (never silence).
        - While agent_busy: audio is sent but transcripts are ignored AND
          a KeepAlive JSON is sent every 8 s so Deepgram never idles out.
        - On any drop: reconnect automatically after 1 s.
        """
        url = (
            f"wss://api.deepgram.com/v1/listen"
            f"?encoding=linear16&sample_rate={RATE}&channels={CHANNELS}"
            f"&endpointing=400&interim_results=true&smart_format=true"
        )
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}

        audio = pyaudio.PyAudio()
        mic   = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                           input=True, frames_per_buffer=CHUNK)

        print(f"[STT] API key loaded: {DEEPGRAM_API_KEY[:8]}... (len={len(DEEPGRAM_API_KEY)})")
        print(f"[STT] URL: {url}")
        while self.is_running:
            print("[STT] Connecting...")
            try:
                async with websockets.connect(
                    url, additional_headers=headers,
                    ping_interval=None, open_timeout=10
                ) as ws:
                    print("[STT] Connected ✓")

                    # Each connection gets its own cancel scope via a shared event
                    stop_evt = asyncio.Event()

                    async def _send():
                        last_ka = time.time()
                        while self.is_running and not stop_evt.is_set():
                            try:
                                frames = mic.get_read_available()
                                if frames > 0:
                                    data = mic.read(frames, exception_on_overflow=False)
                                    if self.agent_busy:
                                        # Send silence bytes of same length — keeps TCP
                                        # connection alive WITHOUT giving Deepgram real
                                        # audio to transcribe (prevents echo loopback)
                                        await ws.send(bytes(len(data)))
                                    else:
                                        await ws.send(data)
                                    last_ka = time.time()

                                # Extra KeepAlive JSON every 8 s as insurance
                                if time.time() - last_ka >= 8.0:
                                    await ws.send(json.dumps({"type": "KeepAlive"}))
                                    last_ka = time.time()

                                await asyncio.sleep(0.01)
                            except Exception as ex:
                                print(f"[STT send error] {ex}")
                                stop_evt.set()
                                break

                    async def _recv():
                        try:
                            async for raw in ws:
                                if not self.is_running or stop_evt.is_set():
                                    break

                                msg        = json.loads(raw)
                                transcript = (msg.get("channel", {})
                                               .get("alternatives", [{}])[0]
                                               .get("transcript", ""))

                                # Ignore transcripts while agent is busy or in cooldown
                                if self.agent_busy:
                                    continue

                                # Extra guard: ignore for 300 ms after mic re-opens
                                # to flush any Deepgram-buffered echo from speaker
                                if hasattr(self, '_mic_open_at') and                                         (time.time() - self._mic_open_at) < 0.3:
                                    continue

                                if (msg.get("is_final") and
                                        msg.get("speech_final") and
                                        transcript.strip()):

                                    self.t0             = time.time()
                                    self.is_first_audio = True
                                    self.agent_busy     = True

                                    self._append_you(transcript)
                                    self._set_status("🧠 Thinking...", "#00bfff")
                                    asyncio.create_task(self._agent_pipeline(transcript))

                        except Exception as ex:
                            print(f"[STT recv error] {ex}")
                            stop_evt.set()

                    await asyncio.gather(_send(), _recv())

            except Exception as e:
                print(f"[STT] Disconnected ({e}), reconnecting in 1 s...")

            if self.is_running:
                await asyncio.sleep(1)

        mic.stop_stream(); mic.close(); audio.terminate()

    # ── Agent Pipeline ─────────────────────────────────────────────────
    async def _agent_pipeline(self, user_text):
        self._set_status("🧠 Thinking + Working...", "#00bfff")
        try:
            reply, self.conv_history = await run_agent_async(user_text, self.conv_history)

            # Show full reply at once — clean line gap between bubbles
            self._append_agent(reply)

            # Feed sentences to TTS queue
            self._set_status("🔊 Speaking...", "#ffaa00")
            for sentence in re.split(r'(?<=[.!?])\s+', reply):
                if sentence.strip():
                    await self.tts_queue.put(sentence.strip())
            await self.tts_queue.put("LLM_DONE")

        except Exception as e:
            err = f"Sorry, I ran into an error: {e}"
            print(f"[AGENT ERROR] {e}")
            self._append_agent(err)
            await self.tts_queue.put(err)
            await self.tts_queue.put("LLM_DONE")

    # ── WebSocket 2 — Deepgram TTS ─────────────────────────────────────
    async def _tts_websocket(self):
        url = (
            "wss://api.deepgram.com/v1/speak"
            f"?model=aura-asteria-en&encoding=linear16&sample_rate={RATE}"
        )
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}

        while self.is_running:
            try:
                async with websockets.connect(
                    url, additional_headers=headers, ping_interval=None
                ) as ws:

                    async def _send():
                        while self.is_running:
                            text = await self.tts_queue.get()
                            if text == "LLM_DONE":
                                await ws.send(json.dumps({"type": "Flush"}))
                            else:
                                await ws.send(json.dumps({"type": "Speak", "text": text + " "}))

                    async def _recv():
                        async for msg in ws:
                            if not self.is_running:
                                break
                            if isinstance(msg, bytes):
                                if self.is_first_audio and self.t0:
                                    self._set_latency((time.time() - self.t0) * 1000)
                                    self.is_first_audio = False
                                self.audio_player.play(msg)
                            elif isinstance(msg, str):
                                if json.loads(msg).get("type") == "Flushed":
                                    self.audio_player.unlock()

                    await asyncio.gather(_send(), _recv())

            except Exception as e:
                print(f"[TTS] Reconnecting after error: {e}")
                await asyncio.sleep(1)

# ==========================================
# Entry Point
# ==========================================
if __name__ == "__main__":
    app = VoiceAgentApp()
    app.mainloop()