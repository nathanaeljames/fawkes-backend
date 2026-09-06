from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch
import torchaudio
import numpy as np
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = Path("/root/fawkes/models/coqui_xtts/XTTS-v2/")
CONFIG_PATH = MODEL_DIR / "config.json"
SPEAKERS_DIR = Path("speakers")

# Load model
xtts_config = XttsConfig()
xtts_config.load_json(CONFIG_PATH)
xtts_model = Xtts.init_from_config(xtts_config)
xtts_model.load_checkpoint(
    config=xtts_config,
    checkpoint_dir=MODEL_DIR,
    eval=True
)
xtts_model.to(device)

def save_speaker_representation(wav_path: Path, speaker_id: str):
    # Extract speaker representation
    with torch.no_grad():
        gpt_latent, speaker_embedding = xtts_model.get_conditioning_latents(str(wav_path), 16000)

    # Save both tensors to .pt file
    save_path = SPEAKERS_DIR / f"{speaker_id}.pt"
    torch.save({
        "gpt_cond_latent": gpt_latent.to("cpu"),
        "speaker_embedding": speaker_embedding.to("cpu")
    }, save_path)

    print(f"Saved speaker data for '{speaker_id}' to {save_path}")

if __name__ == "__main__":
    wav_file = Path("/root/fawkes/audio_samples/sofia_vergara.wav")
    save_speaker_representation(wav_file, "sofia_vergara")