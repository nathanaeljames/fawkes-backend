import io
import wave
from piper.voice import PiperVoice

# Load the Piper model
model_path = "/root/models/en_GB-northern_english_male-medium.onnx"
voice = PiperVoice.load(model_path)

# Text to synthesize
text = "This is an example of text-to-speech using Piper TTS."

# Create an in-memory buffer
audio_stream = io.BytesIO()

# Open it as a wave file and configure it correctly
with wave.open(audio_stream, "wb") as wav_file:
    wav_file.setnchannels(1)  # Mono
    wav_file.setsampwidth(2)  # 16-bit PCM
    wav_file.setframerate(voice.config.sample_rate)  # Use Piper's sample rate

    # Synthesize directly into the wave file
    voice.synthesize(text, wav_file)

# Retrieve the raw PCM audio data (excluding WAV headers if needed)
audio_stream.seek(0)
raw_pcm_audio = audio_stream.read()

# Save to a file (optional)
with open("output.wav", "wb") as f:
    f.write(raw_pcm_audio)

print("Audio synthesis successful!")
