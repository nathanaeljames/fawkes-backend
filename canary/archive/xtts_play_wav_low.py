from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch
import torchaudio
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"

start = time.time()
# Load config
xtts_config = XttsConfig()
xtts_config.load_json("/root/fawkes/models/coqui_xtts/XTTS-v2/config.json")
if device=="cuda":
    print("gpu-acceleration is enabled")
print("Loaded config:", time.time() - start)
# Load model
t0 = time.time()
xtts_model = Xtts.init_from_config(xtts_config)
xtts_model.load_checkpoint(
    config=xtts_config,
    checkpoint_dir="/root/fawkes/models/coqui_xtts/XTTS-v2/",
    eval=True
)
xtts_model.to(device)
print("Model loaded:", time.time() - t0)

# Synthesize
t1 = time.time()
outputs = xtts_model.synthesize(
    text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent",
    config=xtts_config,
    speaker_wav='/root/fawkes/audio_samples/neilgaiman_01.wav',
    language="en"
)
print("Synthesis time:", time.time() - t1)

# Save
t2 = time.time()

# Extract waveform and convert to torch tensor
wav = outputs["wav"]
wav_tensor = torch.tensor(wav).unsqueeze(0)  # shape: [1, num_samples]

# Resample to 16kHz if needed
orig_sample_rate = 22050  # XTTS default
target_sample_rate = 16000
if orig_sample_rate != target_sample_rate:
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=target_sample_rate)
    wav_tensor = resampler(wav_tensor)

# Save to file
output_path = "./samples/xtts_output_05.wav"
torchaudio.save(output_path, wav_tensor, target_sample_rate)

print(f"Speech saved to {output_path}")
print("Save time:", time.time() - t2)
print("Total time:", time.time() - start)
