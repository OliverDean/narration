#!/usr/bin/env python3
"""
Single-narrator TTS (speaker p267) for .txt file(s). If input_path is a directory,
it processes all .txt files within. Each .txt becomes a .mp3.
"""

import argparse
import os
import sys
import tempfile
from pydub import AudioSegment
from TTS.api import TTS

AudioSegment.converter = r"C:\Program Files\ffmpeg-7.0.2-full_build\bin\ffmpeg.exe"
AudioSegment.ffmpeg = r"C:\Program Files\ffmpeg-7.0.2-full_build\bin\ffmpeg.exe"

def text_to_mp3(input_file, tts, speaker_id, bitrate="192k"):
    """
    Given a path to a .txt file, read the text, do TTS, and save .mp3.
    """
    # 1) Confirm file exists
    if not os.path.isfile(input_file):
        print(f"[ERROR] File not found: {input_file}")
        return

    # 2) Read entire text
    with open(input_file, "r", encoding="utf-8") as f:
        text_data = f.read().strip()
    if not text_data:
        print(f"[ERROR] Input file is empty: {input_file}")
        return

    # 3) Generate a temporary WAV
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav_path = tmp_wav.name
    tmp_wav.close()

    try:
        tts.tts_to_file(text=text_data, speaker=speaker_id, file_path=tmp_wav_path)
    except Exception as e:
        print(f"[ERROR] TTS generation failed for {input_file}: {e}")
        if os.path.exists(tmp_wav_path):
            os.remove(tmp_wav_path)
        return

    # 4) Compress WAV -> MP3
    output_mp3 = os.path.splitext(input_file)[0] + ".mp3"
    try:
        sound = AudioSegment.from_wav(tmp_wav_path)
        sound.export(output_mp3, format="mp3", bitrate=bitrate)
    except Exception as e:
        print(f"[ERROR] MP3 compression failed for {input_file}: {e}")
        return
    finally:
        if os.path.exists(tmp_wav_path):
            os.remove(tmp_wav_path)

    print(f"[OK] Processed {input_file} -> {output_mp3}")


def main():
    parser = argparse.ArgumentParser(
        description="TTS with speaker p267, no chunking, compressed to MP3. "
                    "Accepts either a single .txt file or a directory of .txt files."
    )
    parser.add_argument("input_path", help="Path to a .txt file OR a directory containing .txt files.")
    parser.add_argument("--model_name", default="tts_models/en/vctk/vits", 
                        help="Coqui TTS model (default: tts_models/en/vctk/vits).")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available.")
    parser.add_argument("--bitrate", default="192k", help="MP3 bitrate (default: 192k).")

    args = parser.parse_args()
    input_path = args.input_path

    # 1) Check if path exists
    if not os.path.exists(input_path):
        print("[ERROR] Specified path does not exist:", input_path)
        sys.exit(1)

    # 2) Load TTS model once
    try:
        tts = TTS(model_name=args.model_name, gpu=args.gpu)
    except Exception as e:
        print("[ERROR] Could not load TTS:", e)
        sys.exit(1)

    # Choose your desired speaker here
    speaker_id = "p267"

    # 3) If directory, process all .txt files; if file, just process that one
    if os.path.isdir(input_path):
        # Get all .txt files in the directory (non-recursive)
        txt_files = [f for f in os.listdir(input_path) if f.lower().endswith(".txt")]
        if not txt_files:
            print("[INFO] No .txt files found in directory:", input_path)
            sys.exit(0)

        for txt_file in txt_files:
            full_path = os.path.join(input_path, txt_file)
            text_to_mp3(full_path, tts, speaker_id, bitrate=args.bitrate)
    else:
        # Single file
        text_to_mp3(input_path, tts, speaker_id, bitrate=args.bitrate)

    print("Done.")

if __name__ == "__main__":
    main()
