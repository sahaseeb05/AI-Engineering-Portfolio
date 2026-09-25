# Live Fact Checker

A local Streamlit dashboard that listens to microphone audio, transcribes it with faster-whisper on CUDA, extracts factual claims with Ollama, and verifies them against live DuckDuckGo snippets.

## Requirements

- Python 3.10+
- NVIDIA GPU with a working CUDA setup for faster-whisper
- Ollama installed and running locally
- A pulled model, for example `ollama pull qwen2.5:7b`
- PortAudio/PyAudio support. On Windows, install a matching PyAudio wheel if a source build fails.

## Setup

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
ollama pull qwen2.5:7b
streamlit run app.py
```

Open the local URL printed by Streamlit, allow microphone access when prompted, and press **Start listening**. Audio is processed locally; only DuckDuckGo receives the extracted claim for search snippets.

## Notes

- The default Whisper model is `small`; set `WHISPER_MODEL=medium` in `.env` when the GPU has enough VRAM.
- Set `OLLAMA_MODEL` to a model already installed in Ollama.
- A transcript or claim can be skipped when the local model, microphone, CUDA runtime, or search service is unavailable; the dashboard reports the error instead of crashing.
