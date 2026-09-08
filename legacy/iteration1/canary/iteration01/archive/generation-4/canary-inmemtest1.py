import torch
import torchaudio
import numpy as np

# ASR model imports
#from nemo.collections.asr.models import SALM
from nemo.collections.speechlm2.models import SALM
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# --- Custom Dataset and DataLoader Components ---

class InmemoryAudioDataset(Dataset):
    """
    A custom PyTorch Dataset that serves audio chunks from an in-memory buffer.
    
    This dataset is designed to work with the NeMo model's inference loop,
    providing chunks of a pre-loaded audio tensor without relying on disk I/O.
    """
    def __init__(self, audio_tensor: torch.Tensor, chunk_size_in_samples: int):
        """
        Initializes the dataset with an audio tensor and chunking parameters.

        Args:
            audio_tensor (torch.Tensor): A PyTorch tensor containing the full audio waveform.
            chunk_size_in_samples (int): The number of samples per audio chunk.
        """
        self.audio_tensor = audio_tensor
        self.chunk_size = chunk_size_in_samples
        # Calculate the number of chunks from the total audio length.
        # This handles the case where the audio length is not a perfect multiple of the chunk size.
        self.num_chunks = (audio_tensor.shape[1] + self.chunk_size - 1) // self.chunk_size

    def __len__(self):
        """Returns the total number of audio chunks."""
        return self.num_chunks

    def __getitem__(self, idx: int):
        """
        Retrieves a single chunk of audio at the specified index.

        Args:
            idx (int): The index of the chunk to retrieve.

        Returns:
            A tuple containing the audio chunk tensor, its length, and a dummy label.
        """
        start = idx * self.chunk_size
        end = min(start + self.chunk_size, self.audio_tensor.shape[1])
        chunk = self.audio_tensor[:, start:end]

        # The NeMo dataloader expects a tuple of (audio_signal, audio_length, label).
        # We provide a dummy label here as it's not used during transcription.
        return (chunk, torch.tensor([chunk.shape[1]], dtype=torch.long), "transcription_placeholder")

def collate_fn(batch):
    """
    A custom collate function for the DataLoader to pad audio chunks to a
    uniform length within a batch. This is necessary for batching with
    variable-length sequences.
    """
    # Unpack the batch into separate lists for audio signals and lengths.
    audio_signals = [item[0].squeeze(0) for item in batch]
    audio_lengths = torch.LongTensor([item[1].item() for item in batch])
    
    # Pad the audio signals to the same length for consistent batch processing.
    padded_audio_signals = pad_sequence(audio_signals, batch_first=True, padding_value=0.0)

    # Return the padded signals and their original lengths.
    return padded_audio_signals, audio_lengths

# --- Main Program Logic ---

def main():
    # Hard-coded file path for the proof of concept.
    PATH_TO_AUDIO_FILE = "samples/xtts_output_02.wav"
    SAMPLE_RATE = 16000
    CHUNK_SECONDS = 40.0

    print("Loading Canary-Qwen model...")
    # Load the model. This is a one-time, expensive operation.
    canary_model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')
    canary_model.eval()
    print("Model loaded successfully.")

    print(f"Loading audio file from {PATH_TO_AUDIO_FILE} into memory...")
    try:
        # Load the entire audio file into an in-memory PyTorch tensor.
        audio_buffer, orig_sample_rate = torchaudio.load(PATH_TO_AUDIO_FILE)
    except FileNotFoundError:
        print(f"Error: The file at {PATH_TO_AUDIO_FILE} was not found.")
        return
    except Exception as e:
        print(f"An error occurred while loading the audio file: {e}")
        return
    
    print(f"Audio file loaded. Original sample rate: {orig_sample_rate} Hz.")

    # Resample and convert to mono if the audio doesn't match the model's requirements.
    if orig_sample_rate != SAMPLE_RATE:
        print(f"Resampling audio from {orig_sample_rate} Hz to {SAMPLE_RATE} Hz...")
        resampler = torchaudio.transforms.Resample(orig_sample_rate, SAMPLE_RATE)
        audio_buffer = resampler(audio_buffer)
    
    # Check for multi-channel audio and convert to mono by taking the mean.
    if audio_buffer.shape[0] > 1:
        print("Converting multi-channel audio to mono...")
        audio_buffer = torch.mean(audio_buffer, dim=0, keepdim=True)

    # Calculate the chunk size in samples.
    chunk_size_samples = int(CHUNK_SECONDS * SAMPLE_RATE)

    print(f"Creating in-memory data loader with chunk size of {CHUNK_SECONDS} seconds...")
    # Create the custom dataset and dataloader.
    # The dataloader will now serve chunks from the in-memory buffer.
    dataset = InmemoryAudioDataset(audio_buffer, chunk_size_samples)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    print("Starting transcription...")
    # The model's transcribe method is called with the custom dataloader.
    # This process will not perform any additional disk I/O.
    with torch.no_grad():
        transcription_hypotheses = canary_model.transcribe(dataloader=dataloader, return_hypotheses=True)

    # Extract the text from the hypotheses. The result is a list of lists of hypotheses.
    # We join them into a single string.
    if transcription_hypotheses:
        full_transcription = " ".join([h.text for h in transcription_hypotheses[0]])
        print("\n--- Transcription Complete ---")
        print(full_transcription)
    else:
        print("Transcription failed or returned no results.")

if __name__ == "__main__":
    main()
