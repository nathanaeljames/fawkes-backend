import torch
import nemo.collections.asr as nemo_asr

# --- 1. Define the path to your .nemo model file ---
nemo_model_path = "/root/fawkes/models/ecapa_tdnn_embed/ecapa_tdnn.nemo"

# --- 2. Restore the model from the local .nemo file ---
speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(nemo_model_path)
print("Model restored successfully.")

# --- 3. Set the model to evaluation mode and move to device ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
speaker_model.eval().to(device)
print(f"Model moved to device: {device}")

# --- 4. Generate a raw audio tensor ---
# This is the input tensor the entire pipeline needs.
sample_rate = 16000
duration_seconds = 5
num_samples = int(sample_rate * duration_seconds)
audio_tensor = torch.randn(1, num_samples).to(device)

# --- 5. Generate the corresponding audio lengths tensor ---
audio_lengths = torch.tensor([num_samples]).to(device)

# --- 6. Pass the raw audio tensor and its lengths through the entire model ---
# The EncDecSpeakerLabelModel's forward method handles the full pipeline internally:
# raw audio -> preprocessor -> encoder -> pooling -> projection.
with torch.no_grad():
    embeddings, _ = speaker_model.forward(
        audio_tensor, audio_lengths
    )

# --- 7. View the output embeddings ---
print("\nInference complete.")
print(f"Final embeddings tensor shape: {embeddings.shape}")
print(f"Final embeddings: {embeddings}")
