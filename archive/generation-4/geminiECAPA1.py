import torchaudio

# ... (rest of your code)

class ECAPAProcessor:
    # ... (rest of your class initialization)

    def extract_embedding_from_file(self, wav_path, sample_rate=None):
        """
        Extract ECAPA embedding from a WAV file.
        This version manually loads the audio and ensures it is mono.
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        try:
            wav_path = str(wav_path)
            print(f"[ECAPA] Extracting embedding from file: {wav_path}")
            
            # Load the audio file as a tensor
            audio_tensor, sr = torchaudio.load(wav_path)

            # Resample audio to 16kHz if needed
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate).to(self.device)
                audio_tensor = resampler(audio_tensor)

            # Convert to mono if the file is multi-channel
            if audio_tensor.shape[0] > 1:
                audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
            
            # Ensure the tensor is 2-dimensional (batch, time) by squeezing
            audio_tensor = audio_tensor.squeeze(0).unsqueeze(0)
            
            # Pass the tensor and its length to the model's get_embedding method
            audio_len = torch.tensor([audio_tensor.shape[1]], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                embeddings = self.model.get_embedding(audio_signal=audio_tensor, length=audio_len)

            embedding_np = embeddings.cpu().numpy().squeeze()
            
            print(f"[ECAPA] Extracted embedding shape: {embedding_np.shape}")
            return embedding_np
        except Exception as e:
            print(f"[ECAPA] Error extracting embedding from file {wav_path}: {e}")
            return None

    def extract_embedding_from_buffer(self, audio_int16, sample_rate=None):
        """
        Extract ECAPA embedding from int16 PCM audio buffer.
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        try:
            print(f"[ECAPA] Extracting embedding from buffer: shape={audio_int16.shape}, sample_rate={sample_rate}")
            
            # Convert int16 to float32 in range [-1, 1]
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            # Convert to torch tensor and add batch dimension
            audio_tensor = torch.from_numpy(audio_float32).unsqueeze(0).to(self.device)
            
            # Squeeze to ensure the tensor is 2-dimensional (batch, time)
            audio_tensor = audio_tensor.squeeze()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            audio_len = torch.tensor([audio_tensor.shape[1]], dtype=torch.long).to(self.device)

            with torch.no_grad():
                embeddings = self.model.get_embedding(audio_signal=audio_tensor, length=audio_len)

            embedding_np = embeddings.cpu().numpy().squeeze()
            
            print(f"[ECAPA] Extracted embedding shape from buffer: {embedding_np.shape}")
            return embedding_np
        except Exception as e:
            print(f"[ECAPA] Error extracting embedding from buffer: {e}")
            return None