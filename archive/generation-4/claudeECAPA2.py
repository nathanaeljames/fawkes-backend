def extract_embedding_from_file(self, wav_path, sample_rate=None):
    """
    Extract ECAPA embedding from a WAV file.
    
    Args:
        wav_path (str or Path): Path to the WAV file
        sample_rate (int, optional): Expected sample rate (defaults to instance sample_rate)
        
    Returns:
        np.ndarray: ECAPA embedding as numpy array, or None on error
    """
    if sample_rate is None:
        sample_rate = self.sample_rate
        
    try:
        wav_path = str(wav_path)
        print(f"[ECAPA] Extracting embedding from file: {wav_path}")
        
        # Load audio file using librosa (handles various formats and resampling)
        audio_data, sr = librosa.load(wav_path, sr=sample_rate, mono=True)
        
        # Convert to torch tensor and add batch dimension
        audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).to(self.device)
        audio_len = torch.tensor([audio_tensor.shape[1]], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            # Use the model's built-in method to extract speaker embeddings
            # This should return the proper fixed-dimension embedding
            embeddings = self.model.get_embedding(input_signal=audio_tensor, input_signal_length=audio_len)
            
            # Convert to numpy array on CPU
            embedding_np = embeddings.cpu().numpy().squeeze()
            
        print(f"[ECAPA] Extracted embedding shape: {embedding_np.shape}")
        return embedding_np
        
    except Exception as e:
        print(f"[ECAPA] Error extracting embedding from file {wav_path}: {e}")
        return None

def extract_embedding_from_buffer(self, audio_int16, sample_rate=None):
    """
    Extract ECAPA embedding from int16 PCM audio buffer.
    
    Args:
        audio_int16 (np.ndarray): Audio data as int16 PCM
        sample_rate (int, optional): Sample rate (defaults to instance sample_rate)
        
    Returns:
        np.ndarray: ECAPA embedding as numpy array, or None on error
    """
    if sample_rate is None:
        sample_rate = self.sample_rate
        
    try:
        print(f"[ECAPA] Extracting embedding from buffer: shape={audio_int16.shape}, sample_rate={sample_rate}")
        
        # Convert int16 to float32 in range [-1, 1]
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        
        # Convert to torch tensor and add batch dimension
        audio_tensor = torch.from_numpy(audio_float32).unsqueeze(0).to(self.device)
        audio_len = torch.tensor([audio_tensor.shape[1]], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            # Use the model's built-in method to extract speaker embeddings
            # This should return the proper fixed-dimension embedding
            embeddings = self.model.get_embedding(input_signal=audio_tensor, input_signal_length=audio_len)
            
            # Convert to numpy array on CPU
            embedding_np = embeddings.cpu().numpy().squeeze()
            
        print(f"[ECAPA] Extracted embedding shape from buffer: {embedding_np.shape}")
        return embedding_np
        
    except Exception as e:
        print(f"[ECAPA] Error extracting embedding from buffer: {e}")
        return None