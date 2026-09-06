# Check piper-tts version
import piper
print(f"Piper version: {piper.__version__ if hasattr(piper, '__version__') else 'Unknown'}")

# Check installed package version via pip
import subprocess
result = subprocess.run(['pip', 'show', 'piper-tts'], capture_output=True, text=True)
print(result.stdout)

# Check the actual PiperVoice methods
from piper.voice import PiperVoice
print("PiperVoice methods:", [m for m in dir(PiperVoice) if not m.startswith('_')])

# Load a voice and check its methods
voice = PiperVoice.load('/path/to/your/model.onnx')  # Use your actual model path
print("Voice instance methods:", [m for m in dir(voice) if not m.startswith('_')])

# Test if synthesize_stream_raw exists
if hasattr(voice, 'synthesize_stream_raw'):
    print("✓ synthesize_stream_raw exists")
else:
    print("✗ synthesize_stream_raw does NOT exist")