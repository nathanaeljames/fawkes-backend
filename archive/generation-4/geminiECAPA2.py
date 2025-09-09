import torchaudio
import torch

# ... (rest of your code)

class ECAPAProcessor:
    # ... (rest of your class initialization)

    def extract_embedding_from_file(self, wav_path):
        """
        Extract ECAPA embedding from a WAV file using the model's built-in file handling.
        """
        try:
            print(f"[ECAPA] Extracting embedding from file: {wav_path}")
            
            # The model's get_embedding method is designed to handle file paths.
            with torch.no_grad():
                embeddings = self.model.get_embedding(wav_path=wav_path)
            
            embedding_np = embeddings.cpu().numpy().squeeze()
            
            print(f"[ECAPA] Extracted embedding shape: {embedding_np.shape}")
            return embedding_np
        except Exception as e:
            print(f"[ECAPA] Error extracting embedding from file {wav_path}: {e}")
            return None

    def extract_embedding_from_buffer(self, audio_int16, sample_rate=None):
        """
        Extract ECAPA embedding from int16 PCM audio buffer by manually pre-processing.
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        try:
            print(f"[ECAPA] Extracting embedding from buffer: shape={audio_int16.shape}, sample_rate={sample_rate}")
            
            # Convert int16 to float32 in range [-1, 1]
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            # Convert to torch tensor, ensure 2-dimensional, and move to device
            audio_tensor = torch.from_numpy(audio_float32).unsqueeze(0).to(self.device)
            if audio_tensor.dim() > 2:
                audio_tensor = audio_tensor.squeeze()
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
            
            audio_len = torch.tensor([audio_tensor.shape[1]], dtype=torch.long).to(self.device)

            with torch.no_grad():
                # Manually pre-process the audio tensor to get features
                processed_features, processed_len = self.model.preprocessor(
                    input_signal=audio_tensor, length=audio_len
                )
                
                # Pass the features to the model's encoder to get the final embedding
                embeddings = self.model.encoder(processed_features)
                
            embedding_np = embeddings.cpu().numpy().squeeze()
            
            print(f"[ECAPA] Extracted embedding shape from buffer: {embedding_np.shape}")
            return embedding_np
        except Exception as e:
            print(f"[ECAPA] Error extracting embedding from buffer: {e}")
            return None
