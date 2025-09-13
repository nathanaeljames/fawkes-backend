from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# generate speech by cloning a voice using default settings
#It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.
tts.tts_to_file(text="The beige hue on the waters of the loch impressed all, including the French queen, before she heard that symphony again, just as young Arthur wanted.",
                file_path="xtts_output_02.wav",
                speaker_wav="/root/fawkes/audio_samples/neilgaiman_01.wav",
                language="en")

