import torch
import torchaudio
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf

def extract_embedding_from_file(audio_file_path: str, model_path: str):
    """
    Loads a .nemo model and a .wav file, preprocesses the audio into a tensor,
    and extracts a speaker embedding directly from the tensor in memory.

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

    # --- 2. Load the audio file as a raw tensor ---
    print(f"Loading audio from: {audio_file_path}")
    try:
        waveform, sample_rate = torchaudio.load(audio_file_path)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        print("Please ensure the audio file exists and is a valid WAV file.")
        return

    # --- 3. Preprocess the audio using the model's built-in featurizer ---
    # The NeMo model expects specific features (e.g., Mel-spectrograms).
    # We use the model's internal processing module to correctly transform
    # the raw audio tensor into the required features.
    print("Preprocessing audio tensor...")
    with torch.no_grad():
        # The model's audio processor requires a length tensor
        audio_len = torch.tensor([waveform.shape[1]], dtype=torch.long, device=model.device)
        processed_signal, processed_signal_len = model.audio_preprocessing(
            input_signal=waveform.to(model.device),
            length=audio_len.to(model.device)
        )

        # --- 4. Extract the embedding from the processed tensor ---
        # The 'extract_embedding' method takes the preprocessed features directly.
        print("Extracting embedding...")
        embedding = model.extract_embedding(
            input_signal=processed_signal,
            length=processed_signal_len
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
    extract_embedding_from_file(WAV_FILE_PATH, MODEL_FILE_PATH)
