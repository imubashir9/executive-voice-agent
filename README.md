# 🤖 Royal Cyber Executive Voice Agent

A real-time, conversational AI executive assistant built with Python. This application features a stutter-free voice interface and integrates with multiple services to manage scheduling, communications, and CRM data. 

## ✨ Features

* **Continuous Voice Interface:** Utilizes Deepgram for real-time, dual-WebSocket Speech-to-Text (STT) and Text-to-Speech (TTS) with smooth audio playback.
* **Intelligent Agent:** Powered by Azure OpenAI (GPT-4o-mini) to naturally converse and execute tools based on user requests.
* **Calendar Management:** Integrates with Cal.com (v2) to list event types, check availability, book meetings, and cancel appointments.
* **Omnichannel Messaging:** Sends SMS and WhatsApp messages via Twilio, and emails via standard SMTP.
* **Local CRM:** Creates leads (`leads.json`) and logs call summaries (`call_logs.txt`) directly to local files.
* **Custom UI:** Includes a dark-themed graphical user interface built with CustomTkinter for monitoring agent status, response latency, and conversation history.

## 🛠️ Prerequisites

Before you begin, ensure you have the following installed:
* Python 3.8+
* [PyAudio](https://pypi.org/project/PyAudio/) (may require system-level audio libraries depending on your OS)

You will also need API keys for the following services:
* Deepgram
* Azure OpenAI
* Cal.com
* Twilio
* Gmail (App Password required for SMTP)

## 🚀 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone [https://github.com/your-username/executive-voice-agent.git](https://github.com/your-username/executive-voice-agent.git)
   cd executive-voice-agent

2. **Install dependencies**
   Install the required Python packages:
   ```bash
   pip install customtkinter websockets pyaudio openai twilio python-dotenv httpx python-dateutil
   ```

3. **Environment Variables**
   Create a `.env` file in the root directory (make sure it is added to your `.gitignore`) and add your credentials. See the `.env.example` file for the required format.

## 💻 Usage

Run the application from your terminal:

```bash
python voice-ui-agent.py
```

Once the UI launches:
1. Click **🎙️ Start Listening** to initialize the connection.
2. Speak naturally into your microphone. The agent will transcribe your speech, process the intent, execute any necessary tools, and reply audibly.
3. Click **⏹ Stop** to end the session.

## ⚠️ Security Note
Never commit your `.env` file to version control. Always keep your API keys and passwords secure.
```
