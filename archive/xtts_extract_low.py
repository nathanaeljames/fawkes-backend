from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch
import torchaudio
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
audio_path = "/root/fawkes/audio_samples/neilgaiman_01.wav"
embedding_path = "neilgaiman.npy"

# Load model
xtts_config = XttsConfig()
xtts_config.load_json("/root/fawkes/models/coqui_xtts/XTTS-v2/config.json")
xtts_model = Xtts.init_from_config(xtts_config)
xtts_model.load_checkpoint(
    config=xtts_config,
    checkpoint_dir="/root/fawkes/models/coqui_xtts/XTTS-v2/",
    eval=True
)
xtts_model.to(device)

def prepare_audio(path, target_sr=22050):
    waveform, sr = torchaudio.load(path)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform

waveform = prepare_audio(audio_path)
embedding = xtts_model.get_speaker_embedding(waveform, sr=22050)

# Save it
np.save(embedding_path, embedding.cpu().numpy())
print(f"Saved embedding to {embedding_path}")