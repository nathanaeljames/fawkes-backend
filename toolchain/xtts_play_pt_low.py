import torch
import torchaudio
from pathlib import Path
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import numpy as np

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

# Assume xtts_model and device already initialized elsewhere

SPEAKERS_DIR = Path("speakers")


def load_speakers_into_manager():
    """
    Load all .pt files in the speakers directory into speaker_manager.
    """
    for pt_file in SPEAKERS_DIR.glob("*.pt"):
        data = torch.load(pt_file, map_location="cpu")
        xtts_model.speaker_manager.speakers[pt_file.stem] = {
            "gpt_cond_latent": data["gpt_cond_latent"],
            "speaker_embedding": data["speaker_embedding"]
        }
    print(f"Loaded {len(xtts_model.speaker_manager.speakers)} speakers into speaker_manager.")


def synthesize_to_wav(speaker_name, text, output_path):
    """
    Generate speech from text using a named speaker and save as a .wav file.
    """
    speaker_data = xtts_model.speaker_manager.speakers.get(speaker_name)

    if speaker_data is None:
        raise ValueError(f"Speaker '{speaker_name}' not found in speaker_manager.")

    stream = xtts_model.inference_stream(
        text=text,
        language="en",
        gpt_cond_latent=speaker_data["gpt_cond_latent"].to(xtts_model.device),
        speaker_embedding=speaker_data["speaker_embedding"].to(xtts_model.device),
    )

    # Collect waveform chunks into a full tensor
    audio_chunks = [chunk for chunk in stream]
    audio_tensor = torch.cat(audio_chunks, dim=-1).cpu()

    # Save to .wav
    torchaudio.save(output_path, audio_tensor.unsqueeze(0), 24000)
    print(f"Saved synthesized audio to {output_path}")

if __name__ == "__main__":
    #load all speakers one time into dictionary
    load_speakers_into_manager()
    #compute and store output
    synthesize_to_wav('neil_gaiman', "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent", output_path="samples/xtts_from_pt.wav")