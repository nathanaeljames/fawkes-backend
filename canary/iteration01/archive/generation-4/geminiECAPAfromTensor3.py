import torch
import torchaudio
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf

def extract_embedding_from_tensor(audio_waveform: torch.Tensor, model_path: str):
    """
    Loads a .nemo model and extracts a speaker embedding from a PyTorch tensor.
    This function uses the model's forward() method to get the final fixed-size
    embedding, which is the intended way to use the model with tensor inputs.

    Args:
        audio_waveform (torch.Tensor): The raw audio waveform as a PyTorch tensor.
                                       Expected shape: [channels, time_steps].
        model_path (str): Path to the pre-trained .nemo model.
    """
    # --- 1. Load the model ---
    print(f"Loading model from: {model_path}")
    try:
        model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(restore_path=model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model file exists and is a valid NeMo .nemo file.")
        return

    model.eval()

    # --- 2. Prepare the audio tensor for the model ---
    print(f"Preparing audio tensor with shape: {audio_waveform.shape}")

    # Ensure the audio tensor is mono and has a batch dimension.
    # The expected shape for the model is [B, T].
    if audio_waveform.dim() > 1 and audio_waveform.shape[0] > 1:
        print("Warning: Input is multi-channel. Selecting the first channel.")
        audio_waveform = audio_waveform[0].unsqueeze(0)
    elif audio_waveform.dim() == 1:
        audio_waveform = audio_waveform.unsqueeze(0)
    
    audio_waveform = audio_waveform.to(model.device)
    
    # The model's forward() method requires a length tensor
    audio_len = torch.tensor([audio_waveform.shape[1]], dtype=torch.long, device=model.device)

    # --- 3. Pass through the model's forward method ---
    print("Extracting embedding from raw audio tensor...")
    with torch.no_grad():
        # The forward() method returns a tuple. The first element is the embedding.
        _, embedding = model.forward(input_signal=audio_waveform, input_signal_length=audio_len)

    # --- 4. Print the resulting embedding ---
    print("\n--- Embedding Extracted ---")
    print(f"Shape of embedding tensor: {embedding.shape}")
    print(f"Embedding values:\n{embedding}")
    print("----------------------------")

if __name__ == "__main__":
    # Define the model path and the audio tensor
    MODEL_FILE_PATH = "/root/fawkes/models/ecapa_tdnn_embed/ecapa_tdnn.nemo"
    
    # Simulate loading an audio tensor from a file for demonstration
    # In your use case, this tensor would be pre-loaded from memory
    try:
        audio_tensor, sample_rate = torchaudio.load("/root/fawkes/audio_samples/_preprocessed/neilgaiman_01.wav")
        # Ensure the sample rate matches the model's requirements (usually 16000 Hz)
        if sample_rate != 16000:
            print(f"Resampling audio from {sample_rate}Hz to {16000}Hz")
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio_tensor = resampler(audio_tensor)
        extract_embedding_from_tensor(audio_tensor, MODEL_FILE_PATH)
    except Exception as e:
        print(f"Error extracting embedding: {e}")
        print("Please ensure the audio file exists and is a valid WAV file for this example.")
