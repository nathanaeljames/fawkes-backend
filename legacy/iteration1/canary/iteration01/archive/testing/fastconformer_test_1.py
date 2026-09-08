import soundfile as sf
import torch
import numpy as np
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel

# Load the model
asr_model = EncDecHybridRNNTCTCBPEModel.restore_from(
    restore_path="/root/fawkes/models/fc-hybrid-lg-multi/stt_en_fastconformer_hybrid_large_streaming_multi.nemo"
).cuda()
asr_model.eval()

# Load audio
wav_path = "transcribe/16khz_example_speech.wav"
audio, sr = sf.read(wav_path)
assert sr == 16000 and audio.ndim == 1

# Audio buffer parameters
chunk_duration = 0.2  # seconds
buffer_duration = 4.8  # seconds
chunk_size = int(sr * chunk_duration)
buffer_size = int(sr * buffer_duration)

buffer = np.zeros(buffer_size, dtype=np.float32)

print("Transcribing...")

for offset in range(0, len(audio), chunk_size):
    chunk = audio[offset:offset+chunk_size]
    if len(chunk) < chunk_size:
        chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

    # Shift left and append chunk
    buffer = np.roll(buffer, -chunk_size)
    buffer[-chunk_size:] = chunk

    # Extract features manually
    signal_tensor = torch.tensor(buffer, dtype=torch.float32).unsqueeze(0).cuda()
    signal_len = torch.tensor([signal_tensor.shape[-1]]).cuda()

    with torch.no_grad():
        # Perform full forward pass
        logits, predictions = asr_model(input_signal=signal_tensor, input_signal_length=signal_len)

        # Decode the prediction (RNNT decoding)
        hypothesis = asr_model.decoding.decode(predictions)

    partial_text = hypothesis[0].text
    print(f"\r{partial_text}", end="", flush=True)

print("\nDone.")
