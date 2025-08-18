# A simple, standalone Python program to transcribe audio using the Canary-Qwen
# model by feeding an audio tensor directly to the model's generate method.

import torch
import torchaudio
import os
import shutil

# Correct import for the SALM class
from nemo.collections.speechlm2.models import SALM

def main():
    """
    Main function to execute the transcription process.
    """
    # Define the path for the audio file.
    # This script assumes 'samples/xtts_output_02.wav' already exists.
    audio_path = "samples/xtts_output_02.wav"

    # Check if the audio file exists
    if not os.path.exists(audio_path):
        print(f"Error: The audio file '{audio_path}' was not found.")
        print("Please ensure the file exists before running the script.")
        return

    # 1. Download and load the latest version of Canary-Qwen
    print("\nLoading the Canary-Qwen model...")
    # The model is automatically downloaded if not already present in the cache.
    try:
        # Corrected model loading as per your request
        model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Load the audio file and convert it into the proper tensor format
    print(f"Loading audio from '{audio_path}' and converting to tensor...")
    
    # Load the audio file
    # torchaudio.load returns a tuple: (waveform, sample_rate)
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert to 16000 Hz if necessary
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)(waveform)
    
    # Convert to mono-channel if necessary (by taking the mean of all channels)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # Squeeze the channel dimension to get a (time,) tensor and prepare for batching
    audio_tensor = waveform.squeeze(0)
    
    # The model's generate method expects a batch.
    # The audio tensor shape should be (batch_size, time)
    # The audio_lens tensor should be (batch_size,)
    audios = audio_tensor.unsqueeze(0).to(model.device)  # Add a batch dimension
    audio_lens = torch.tensor([audios.shape[1]], dtype=torch.int64).to(model.device)

    # 3. Feed the tensor directly into the model's generate method
    print("Feeding the audio tensor directly into the model to generate transcription...")
    
    # Prepare the prompt, which includes the model's audio locator tag
    # This tells the model where to insert the audio information
    prompts = [
        [{"role": "user", "content": f"Transcribe the following: {model.audio_locator_tag}"}]
    ]

    # Generate the transcription. This happens without any disk I/O for the audio.
    # We'll set max_new_tokens for a fixed length output.
    raw_output_ids = model.generate(
        prompts=prompts,
        audios=audios,
        audio_lens=audio_lens,
        max_new_tokens=128
    )

    # 4. Print the raw model output and the final transcription
    print("\n--- Model Output ---")
    print("Raw token IDs:", raw_output_ids)
    
    # The model's `generate` method returns the token IDs.
    # We can use the model's tokenizer to decode these IDs into human-readable text.
    # Corrected method to use ids_to_text as per the tokenizer's API
    final_transcription = model.tokenizer.ids_to_text(raw_output_ids[0].cpu())

    print("\n--- Final Transcription ---")
    print(f"Transcription: {final_transcription}")

if __name__ == "__main__":
    main()
