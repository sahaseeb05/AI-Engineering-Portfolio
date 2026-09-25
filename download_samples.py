"""Download small, public media samples for local detector smoke tests.

The URLs below point to public repository/demo assets. They are intended for
testing only; the labels are dataset/demo labels and are not forensic truth.
"""

from __future__ import annotations

import os
from pathlib import Path

import requests


BASE_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = BASE_DIR / "test_samples"

SAMPLES = {
    "audio": [
        (
            "real_environment.wav",
            "https://raw.githubusercontent.com/karoldvl/ESC-50/master/audio/1-100032-A-0.wav",
        ),
        (
            "fake_synthetic_voice.wav",
            "https://github.com/CorentinJ/Real-Time-Voice-Cloning/raw/master/demo/obama.wav",
        ),
    ],
    "image": [
        (
            "real_photo.jpg",
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
        ),
        (
            "fake_ai_generated.png",
            "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/txt2img/7.png",
        ),
    ],
    "video": [
        (
            "real_video.mp4",
            "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
        ),
        (
            "fake_face_swap.mp4",
            "https://raw.githubusercontent.com/Dodoression/deepfake-video-cnn-recognition/main/data/fake/1.mp4",
        ),
    ],
}


def download_file(url: str, destination: Path) -> bool:
    """Stream one URL to disk and return whether it completed successfully."""
    print(f"  Downloading {destination.name}...")
    temporary_path = destination.with_suffix(destination.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=60, headers={"User-Agent": "deepfake-detector-samples/1.0"}) as response:
            response.raise_for_status()
            total_bytes = int(response.headers.get("content-length", 0))
            downloaded_bytes = 0
            with temporary_path.open("wb") as output_file:
                for chunk in response.iter_content(chunk_size=1024 * 64):
                    if not chunk:
                        continue
                    output_file.write(chunk)
                    downloaded_bytes += len(chunk)
                    if total_bytes:
                        progress = downloaded_bytes / total_bytes * 100
                        print(f"\r    Progress: {progress:6.1f}%", end="", flush=True)
        if total_bytes and downloaded_bytes != total_bytes:
            raise IOError("the server returned an incomplete file")
        os.replace(temporary_path, destination)
        print(f"\r    Saved: {destination}                 ")
        return True
    except (OSError, requests.RequestException) as error:
        print(f"\n    Failed: {error}")
        if temporary_path.exists():
            temporary_path.unlink()
        return False


def main() -> int:
    """Create the sample tree and download every configured test asset."""
    print(f"Preparing sample directory: {SAMPLES_DIR}")
    failed_downloads = 0
    for media_type, samples in SAMPLES.items():
        media_directory = SAMPLES_DIR / media_type
        os.makedirs(media_directory, exist_ok=True)
        print(f"\n[{media_type.upper()}]")
        for filename, url in samples:
            destination = media_directory / filename
            if destination.exists() and destination.stat().st_size > 0:
                print(f"  Already exists: {destination}")
                continue
            if not download_file(url, destination):
                failed_downloads += 1

    if failed_downloads:
        print(f"\nSample download finished with {failed_downloads} failure(s).")
        return 1
    print("\nSample download complete.")
    print(f"Files are available under: {SAMPLES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())