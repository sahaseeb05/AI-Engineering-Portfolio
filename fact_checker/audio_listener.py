import queue
import threading
import time
import numpy as np
import pyaudio
from faster_whisper import WhisperModel

class AudioListener:
    def __init__(self, model_size="small", sample_rate=16000, chunk_duration=3):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.transcripts = queue.Queue()
        self.is_running = False
        self.thread = None

        # Forced CPU model execution to avoid CUDA/cublas DLL errors
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def _audio_callback(self):
        p = pyaudio.PyAudio()
        stream = None
        
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=1024
            )
        except Exception as e:
            print(f"Mic Access Error: {e}")
            self.transcripts.put(f"[Mic Access Error: Check permissions/device]")
            p.terminate()
            return

        audio_buffer = []

        while self.is_running:
            try:
                data = stream.read(1024, exception_on_overflow=False)
                if data:
                    audio_buffer.append(np.frombuffer(data, dtype=np.int16))

                total_samples = sum(len(chunk) for chunk in audio_buffer)
                if total_samples >= self.chunk_size:
                    full_chunk = np.concatenate(audio_buffer)
                    audio_buffer = []

                    # Noise Gate: Ignore absolute silence to save CPU
                    max_amplitude = np.max(np.abs(full_chunk)) if len(full_chunk) > 0 else 0
                    if max_amplitude < 200:
                        continue

                    # Convert int16 PCM to float32
                    audio_float = full_chunk.astype(np.float32) / 32768.0

                    # Transcribe audio buffer
                    segments, _ = self.model.transcribe(
                        audio_float, 
                        beam_size=1, 
                        vad_filter=True
                    )
                    text = " ".join([segment.text for segment in segments]).strip()

                    if text:
                        print(f"Captured: {text}")
                        self.transcripts.put(text)
            except Exception as e:
                print(f"Read Loop Error: {e}")
                break

        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._audio_callback, daemon=True)
            self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)