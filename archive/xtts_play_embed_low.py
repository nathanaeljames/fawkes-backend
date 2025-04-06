from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch
import torchaudio
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

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

# Load precomputed speaker embedding (.npy)
embedding_np = np.load("neilgaiman.npy")  # shape (512,) or (1,512)
embedding_tensor = torch.from_numpy(embedding_np).float().unsqueeze(0).to(device)

# Generate GPT latent conditioning from text and language
text = "This is a test using a precomputed speaker embedding."
language = "en"

gpt_cond_latent = xtts_model.get_gpt_cond_latents(text, language)

# Now run inference
output_dict = xtts_model.inference(
    text=text,
    language=language,
    gpt_cond_latent=gpt_cond_latent,
    speaker_embedding=embedding_tensor,
)

# Save the waveform
output_waveform = output_dict["wav"]
torchaudio.save("samples/xtts_from_embed.wav", output_waveform.unsqueeze(0).cpu(), 16000)

print("Done. Audio saved.")