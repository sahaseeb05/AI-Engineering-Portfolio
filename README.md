# Multi-Modal Deepfake & Synthetic Media Detector

A Streamlit dashboard for exploring audio, image, video, and live PCM signal indicators associated with synthetic media.

## Setup

```powershell
cd deepfake_detector_system
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The dashboard uses transparent signal-processing heuristics for demonstration and triage. Scores are not forensic proof and should not be used as the sole basis for moderation, identity, employment, legal, or security decisions.

## Modules

- `modules/audio_engine.py`: Mel-spectrogram/MFCC extraction and synthetic-voice heuristic.
- `modules/image_engine.py`: FFT magnitude and edge-noise analysis.
- `modules/video_engine.py`: Sampled frame consistency analysis.
- `modules/live_audio_engine.py`: Raw PCM chunk scoring for virtual audio streams.
