#!/usr/bin/env python3

import torch
import torchaudio
from nemo.collections.asr.models import EncDecSpeakerLabelModel

def main():
    # Load the ECAPA-TDNN model
    model_path = "/root/fawkes/models/ecapa_tdnn_embed/ecapa_tdnn.nemo"
    model = EncDecSpeakerLabelModel.restore_from(model_path)
    model.eval()
    
    # Load the audio file
    audio_path = "./utterances/utterance_badf5312-e8a3-4f91-8279-ef490b4235bb.wav"
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Ensure single channel (mono)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary (ECAPA-TDNN typically expects 16kHz)
    target_sr = 16000
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
    
    print(f"Audio tensor shape: {waveform.shape}")
    print(f"Sample rate: {target_sr}")
    
    # Move tensors to the same device as the model
    device = next(model.parameters()).device
    waveform = waveform.to(device)
    
    # Generate embedding
    with torch.no_grad():
        # NeMo models typically expect audio length in samples as second parameter
        audio_length = torch.tensor([waveform.shape[1]], dtype=torch.long).to(device)
        _, embedding = model.forward(waveform, audio_length)
    
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding: {embedding}")

if __name__ == "__main__":
    main()