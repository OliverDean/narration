from TTS.api import TTS

tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

tts.tts_to_file(
    text="Hello world, this is a test.",
    file_path="test.wav"
)
