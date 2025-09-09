#!/usr/bin/env python3

import torch
import torchaudio
import duckdb
import numpy as np
from nemo.collections.asr.models import EncDecSpeakerLabelModel

def main():
    # Load the ECAPA-TDNN model
    model_path = "/root/fawkes/models/ecapa_tdnn_embed/ecapa_tdnn.nemo"
    model = EncDecSpeakerLabelModel.restore_from(model_path)
    model.eval()
    
    # Load the audio file
    #audio_path = "./utterances/utterance_badf5312-e8a3-4f91-8279-ef490b4235bb.wav"
    audio_path = "/root/fawkes/audio_samples/_preprocessed/nathanael_01.wav"
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
    
    # Connect to DuckDB and compare against stored embeddings
    conn = duckdb.connect('./speakers/speakers.duckdb')
    
    # Query all speakers with their ECAPA embeddings
    query = "SELECT uid, firstname, surname, ecapa_embedding FROM speakers WHERE ecapa_embedding IS NOT NULL"
    results = conn.execute(query).fetchall()
    
    if not results:
        print("No speakers with ECAPA embeddings found in database")
        return
    
    # Convert current embedding to numpy for cosine similarity calculation
    current_embedding = embedding.cpu().numpy().flatten()
    current_embedding = current_embedding / np.linalg.norm(current_embedding)  # Normalize
    
    best_similarity = -1
    best_match = None
    
    print(f"\nComparing against {len(results)} speakers:")
    
    for uid, firstname, surname, stored_embedding in results:
        # Convert stored embedding to numpy
        stored_array = np.array(stored_embedding)
        stored_array = stored_array / np.linalg.norm(stored_array)  # Normalize
        
        # Calculate cosine similarity
        similarity = np.dot(current_embedding, stored_array)
        
        # Format name
        name = f"{firstname}_{surname}" if surname else firstname
        print(f"  {name} (UID: {uid}): {similarity:.4f}")
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = (uid, name)
    
    if best_match:
        print(f"\nBest match: {best_match[1]} (UID: {best_match[0]}) with similarity: {best_similarity:.4f}")
    
    conn.close()

if __name__ == "__main__":
    main()