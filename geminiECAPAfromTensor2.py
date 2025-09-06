import torch
import torchaudio
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf

def extract_embedding_from_tensor(audio_file_path: str, model_path: str):
    """
    Loads a .nemo model and a .wav file, preprocesses the raw audio tensor,
    and extracts a speaker embedding from the features in memory.

    Args:
        audio_file_path (str): Path to the input WAV audio file.
        model_path (str): Path to the pre-trained .nemo model.
    """
    # --- 1. Load the model and its configuration ---
    print(f"Loading model from: {model_path}")
    try:
        model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(restore_path=model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model file exists and is a valid NeMo .nemo file.")
        return

    model.eval()

    # --- 2. Load the audio file into a raw tensor ---
    print(f"Loading audio from: {audio_file_path} into a tensor")
    try:
        # Load the raw audio waveform
        waveform, sample_rate = torchaudio.load(audio_file_path)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        print("Please ensure the audio file exists and is a valid WAV file.")
        return

    # Check and adjust sample rate if necessary
    if sample_rate != model.preprocessor._sample_rate:
        print(f"Resampling audio from {sample_rate}Hz to {model.preprocessor._sample_rate}Hz")
        transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=model.preprocessor._sample_rate)
        waveform = transform(waveform)
        sample_rate = model.preprocessor._sample_rate

    # The model expects a batch of signals, so we unsqueeze to add a batch dimension.
    # The shape should be [B, T], where B is batch size and T is time steps.
    waveform = waveform.to(model.device)
    
    # The model's audio preprocessor requires a length tensor
    audio_len = torch.tensor([waveform.shape[1]], dtype=torch.long, device=model.device)

    # --- 3. Preprocess the audio using the model's built-in preprocessor ---
    # This module takes the raw audio tensor and returns the features (e.g., Mel-spectrograms).
    print("Preprocessing audio tensor using model's internal preprocessor...")
    with torch.no_grad():
        processed_signal, processed_signal_len = model.preprocessor(
            input_signal=waveform,
            length=audio_len
        )
        
        # --- 4. Extract the embedding from the preprocessed features ---
        # The forward method takes the preprocessed features and returns the embedding.
        print("Extracting embedding from processed features...")
        embedding = model(
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_len
        )

    # --- 5. Print the resulting embedding ---
    print("\n--- Embedding Extracted ---")
    print(f"Shape of embedding tensor: {embedding.shape}")
    print(f"Embedding values:\n{embedding}")
    print("----------------------------")

if __name__ == "__main__":
    # Define the file paths as provided in the request
    WAV_FILE_PATH = "/root/fawkes/audio_samples/_preprocessed/neilgaiman_01.wav"
    MODEL_FILE_PATH = "/root/fawkes/models/ecapa_tdnn_embed/ecapa_tdnn.nemo"

    # Execute the main function
    extract_embedding_from_tensor(WAV_FILE_PATH, MODEL_FILE_PATH)
