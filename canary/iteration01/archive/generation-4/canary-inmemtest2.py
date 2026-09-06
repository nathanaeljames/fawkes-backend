import torch
import torchaudio
import numpy as np

# ASR model imports
#from nemo.collections.asr.models import SALM
from nemo.collections.speechlm2.models import SALM
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# --- Custom Dataset and DataLoader Components ---

# The custom Dataset and DataLoader are no longer needed for this approach.
# The SALM model's generate method works directly with the audio tensor
# and a structured prompt, as shown in the updated main function.

# --- Main Program Logic ---

def main():
    # Hard-coded file path for the proof of concept.
    PATH_TO_AUDIO_FILE = "samples/xtts_output_02.wav"
    SAMPLE_RATE = 16000
    
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

    print("Starting transcription...")
    # The SALM model uses a `generate` method that requires a list of prompts.
    # The prompt structure includes a text-based instruction and the audio data itself.
    # We pass the full in-memory audio buffer directly to the `generate` method.
    try:
        with torch.no_grad():
            transcription_hypotheses = canary_model.generate(
                prompts=[[
                    {"role": "user", "content": "Transcribe the following audio:", "audio": audio_buffer}
                ]],
                max_new_tokens=256,
            )
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        return

    # The output of `generate` is a list of lists of token IDs.
    # We need to decode the token IDs back into text.
    if transcription_hypotheses:
        # Assuming a single prompt and single generated response for this use case.
        text_ids = transcription_hypotheses[0]
        full_transcription = canary_model.tokenizer.ids_to_text(text_ids)
        print("\n--- Transcription Complete ---")
        print(full_transcription)
    else:
        print("Transcription failed or returned no results.")

if __name__ == "__main__":
    main()
