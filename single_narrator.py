#!/usr/bin/env python3
"""
Single-narrator TTS (speaker p335) for an entire text file (no chunking),
then compresses to MP3, minimal code/output.
"""

import argparse
import os
import sys
import tempfile
from pydub import AudioSegment
from TTS.api import TTS

AudioSegment.converter = r"C:\Program Files\ffmpeg-7.0.2-full_build\bin\ffmpeg.exe"

AudioSegment.ffmpeg = r"C:\Program Files\ffmpeg-7.0.2-full_build\bin\ffmpeg.exe"

def main():
    parser = argparse.ArgumentParser(description="Single narrator TTS with speaker p335, no chunking, compressed to MP3.")
    parser.add_argument("input_file", help="Path to a .txt file")
    parser.add_argument("--model_name", default="tts_models/en/vctk/vits", help="Coqui TTS model (default: vctk/vits).")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available.")
    parser.add_argument("--bitrate", default="192k", help="MP3 bitrate (default: 192k).")
    args = parser.parse_args()

    # 1) Check file
    if not os.path.isfile(args.input_file):
        print("[ERROR] File not found:", args.input_file)
        sys.exit(1)

    # 2) Read entire text
    with open(args.input_file, "r", encoding="utf-8") as f:
        text_data = f.read().strip()
    if not text_data:
        print("[ERROR] Input file is empty.")
        sys.exit(1)

    # 3) Load TTS model
    try:
        tts = TTS(model_name=args.model_name, gpu=args.gpu)
    except Exception as e:
        print("[ERROR] Could not load TTS:", e)
        sys.exit(1)

    speaker_id = "p267"

    # 4) Generate a single WAV
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav_path = tmp_wav.name
    tmp_wav.close()

    try:
        tts.tts_to_file(text=text_data, speaker=speaker_id, file_path=tmp_wav_path)
    except Exception as e:
        print("[ERROR] TTS generation failed:", e)
        if os.path.exists(tmp_wav_path):
            os.remove(tmp_wav_path)
        sys.exit(1)

    # 5) Compress WAV -> MP3
    output_mp3 = os.path.splitext(args.input_file)[0] + ".mp3"
    try:
        sound = AudioSegment.from_wav(tmp_wav_path)
        sound.export(output_mp3, format="mp3", bitrate=args.bitrate)
    except Exception as e:
        print("[ERROR] MP3 compression failed:", e)
        sys.exit(1)

    # 6) Cleanup & Done
    if os.path.exists(tmp_wav_path):
        os.remove(tmp_wav_path)

    print("Done.")

if __name__ == "__main__":
    main()
