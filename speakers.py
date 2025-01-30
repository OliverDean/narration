import wave
import os
import tempfile
from TTS.api import TTS

# The list of multi-speaker IDs you want to sample:
all_speakers = [
    "p225", "p226", "p227", "p228", "p229", "p230", "p231", "p232", "p233", "p234",
    "p236", "p237", "p238", "p239", "p240", "p241", "p243", "p244", "p245", "p246",
    "p247", "p248", "p249", "p250", "p251", "p252", "p253", "p254", "p255", "p256",
    "p257", "p258", "p259", "p260", "p261", "p262", "p263", "p264", "p265", "p266",
    "p267", "p268", "p269", "p270", "p271", "p272", "p273", "p274", "p275", "p276",
    "p277", "p278", "p279", "p280", "p281", "p282", "p283", "p284", "p285", "p286",
    "p287", "p288", "p292", "p293", "p294", "p295", "p297", "p298", "p299", "p300",
    "p301", "p302", "p303", "p304", "p305", "p306", "p307", "p308", "p310", "p311",
    "p312", "p313", "p314", "p316", "p317", "p318", "p323", "p326", "p329", "p330",
    "p333", "p334", "p335", "p336", "p339", "p340", "p341", "p343", "p345", "p347",
    "p351", "p360", "p361", "p362", "p363", "p364", "p374", "p376"
]

# Initialize your multi-speaker TTS model (vits for en/vctk)
tts = TTS(model_name="tts_models/en/vctk/vits")

print("Available speakers in model:", tts.speakers)
print("We will generate lines for each ID in all_speakers and combine into voices.wav")

combined_data = b""  # We'll accumulate raw PCM data here
first_params = None  # We'll store the audio params from the first chunk

def essential_params(p):
    """
    Return (nchannels, sampwidth, framerate, comptype) ignoring nframes & compname,
    so we only compare the actual audio format.
    """
    return (p.nchannels, p.sampwidth, p.framerate, p.comptype)

temp_files = []  # We'll track temporary WAVs to remove them after combining

try:
    for speaker_id in all_speakers:
        # The line of text you want each speaker to say:
        line_text = f"Hi from speaker {speaker_id}. Hello olivia!"

        # Generate audio to a temporary file
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_wav_name = tmp_wav.name
        tmp_wav.close()

        print(f"[INFO] Generating audio for speaker '{speaker_id}' -> {tmp_wav_name}")
        tts.tts_to_file(text=line_text, speaker=speaker_id, file_path=tmp_wav_name)

        # Open that WAV, read frames, and append to combined_data
        with wave.open(tmp_wav_name, "rb") as wf:
            params = wf.getparams()
            frames = wf.readframes(wf.getnframes())

        if first_params is None:
            # Store first chunk's params
            first_params = params
            combined_data += frames
        else:
            # Compare ignoring length + compname
            if essential_params(params) != essential_params(first_params):
                print(f"[WARN] Speaker '{speaker_id}' has different audio params; skipping.")
            else:
                combined_data += frames

        temp_files.append(tmp_wav_name)

    # Now write combined_data to one final 'voices.wav'
    if combined_data:
        with wave.open("voices.wav", "wb") as wf:
            wf.setparams(first_params)
            wf.writeframes(combined_data)
        print("[INFO] voices.wav created successfully with all speaker lines.")
    else:
        print("[WARN] No valid audio data was collected.")

finally:
    # Remove temporary files
    for tf in temp_files:
        if os.path.exists(tf):
            os.remove(tf)
