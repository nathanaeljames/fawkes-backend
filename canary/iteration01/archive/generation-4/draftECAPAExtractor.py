# ECAPA-TDNN Speaker Recognition Components
# Add these classes and functions to your existing server07d.py

import nemo.collections.asr as nemo_asr
import librosa
import numpy as np
import torch
import asyncio
from pathlib import Path
import json

class ECAPASpeakerEmbedder:
    """
    Wrapper for ECAPA-TDNN speaker verification model.
    Handles loading the model and extracting speaker embeddings from audio.
    """
    
    def __init__(self, model_path, device):
        """
        Initialize the ECAPA-TDNN model.
        
        Args:
            model_path (str): Path to the .nemo model file
            device (str): Device to load the model on ('cuda' or 'cpu')
        """
        print("Pre-loading ECAPA-TDNN speaker embedding model...")
        self.device = device
        self.model_path = model_path
        
        # Load the ECAPA-TDNN model
        self.model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(
            model_path, 
            map_location=torch.device(self.device)
        )
        self.model.eval()
        self.model.to(self.device)
        
        print("ECAPA-TDNN speaker embedding model loaded successfully")
    
    def extract_embedding_from_file(self, wav_path, sample_rate=16000):
        """
        Extract ECAPA embedding from a WAV file.
        
        Args:
            wav_path (str or Path): Path to the WAV file
            sample_rate (int): Expected sample rate (default: 16000)
            
        Returns:
            np.ndarray: ECAPA embedding as numpy array
        """
        try:
            wav_path = str(wav_path)
            print(f"[ECAPA] Extracting embedding from file: {wav_path}")
            
            # Load audio file using librosa (handles various formats and resampling)
            audio_data, sr = librosa.load(wav_path, sr=sample_rate, mono=True)
            
            # Convert to torch tensor and add batch dimension
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).to(self.device)
            audio_len = torch.tensor([audio_tensor.shape[1]], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                # Extract embeddings using the ECAPA model
                embeddings = self.model.forward(input_signal=audio_tensor, input_signal_length=audio_len)
                
                # Convert to numpy array on CPU
                embedding_np = embeddings.cpu().numpy().squeeze()
                
            print(f"[ECAPA] Extracted embedding shape: {embedding_np.shape}")
            return embedding_np
            
        except Exception as e:
            print(f"[ECAPA] Error extracting embedding from file {wav_path}: {e}")
            return None
    
    def extract_embedding_from_buffer(self, audio_int16, sample_rate=16000):
        """
        Extract ECAPA embedding from int16 PCM audio buffer.
        
        Args:
            audio_int16 (np.ndarray): Audio data as int16 PCM
            sample_rate (int): Sample rate of the audio (default: 16000)
            
        Returns:
            np.ndarray: ECAPA embedding as numpy array
        """
        try:
            print(f"[ECAPA] Extracting embedding from buffer: shape={audio_int16.shape}, sample_rate={sample_rate}")
            
            # Convert int16 to float32 in range [-1, 1]
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            # Convert to torch tensor and add batch dimension
            audio_tensor = torch.from_numpy(audio_float32).unsqueeze(0).to(self.device)
            audio_len = torch.tensor([audio_tensor.shape[1]], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                # Extract embeddings using the ECAPA model
                embeddings = self.model.forward(input_signal=audio_tensor, input_signal_length=audio_len)
                
                # Convert to numpy array on CPU
                embedding_np = embeddings.cpu().numpy().squeeze()
                
            print(f"[ECAPA] Extracted embedding shape from buffer: {embedding_np.shape}")
            return embedding_np
            
        except Exception as e:
            print(f"[ECAPA] Error extracting embedding from buffer: {e}")
            return None

async def initial_speaker_imprint(wav_path, firstname, surname=None):
    """
    Extract both XTTS and ECAPA embeddings from a WAV file and store them in the database.
    This replaces the old extract_xtts_embed function with expanded functionality.
    
    Args:
        wav_path (str or Path): Path to the WAV file
        firstname (str): The first name of the speaker
        surname (str, optional): The last name of the speaker. Defaults to None.
    """
    global xtts_wrapper, ecapa_embedder, con
    
    wav_path = Path(wav_path)
    print(f"Creating initial speaker imprint for {firstname} {surname if surname else ''} from {wav_path}...")
    
    try:
        # Extract XTTS embeddings (existing functionality)
        print(f"[Imprint] Extracting XTTS embeddings...")
        with torch.no_grad():
            gpt_cond_latent, speaker_embedding = xtts_wrapper.xtts_model.get_conditioning_latents(str(wav_path), 16000)
        
        print(f"[Imprint] XTTS shapes - GPT: {gpt_cond_latent.shape}, Speaker: {speaker_embedding.shape}")
        
        # Convert XTTS tensors to Python lists (flattened) for DuckDB array storage
        gpt_latent_flat = gpt_cond_latent.cpu().numpy().flatten().tolist()
        xtts_embedding_flat = speaker_embedding.cpu().numpy().flatten().tolist()
        
        # Convert XTTS shapes to JSON strings for storage
        gpt_shape_json = json.dumps(list(gpt_cond_latent.shape))
        xtts_shape_json = json.dumps(list(speaker_embedding.shape))
        
        # Extract ECAPA embedding (new functionality)
        print(f"[Imprint] Extracting ECAPA embedding...")
        ecapa_embedding = ecapa_embedder.extract_embedding_from_file(wav_path, sample_rate=16000)
        
        if ecapa_embedding is None:
            print(f"[Imprint] Failed to extract ECAPA embedding, storing XTTS data only")
            ecapa_embedding_flat = None
        else:
            # Convert ECAPA embedding to Python list for DuckDB array storage
            ecapa_embedding_flat = ecapa_embedding.flatten().tolist()
            print(f"[Imprint] ECAPA embedding shape: {ecapa_embedding.shape}")
        
        # Store all embeddings in database using asyncio thread pool
        def insert_speaker_data():
            con.execute("""
                INSERT INTO speakers 
                (firstname, surname, gpt_cond_latent, gpt_shape, xtts_embedding, xtts_shape, ecapa_embedding) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                firstname, 
                surname, 
                gpt_latent_flat,     # XTTS GPT latent (flattened)
                gpt_shape_json,      # XTTS GPT shape as JSON
                xtts_embedding_flat, # XTTS speaker embedding (flattened)
                xtts_shape_json,     # XTTS speaker shape as JSON
                ecapa_embedding_flat # ECAPA embedding (flattened)
            ))
        
        await asyncio.to_thread(insert_speaker_data)
        
        print(f"[Imprint] Successfully stored complete speaker imprint for {firstname} in DuckDB")
        
        # Reload the XTTS speaker manager to include the new speaker
        xtts_wrapper._load_speakers_from_db()
        
        # Rebuild the ECAPA matcher embedding matrix to include the new speaker
        ecapa_matcher.rebuild_embedding_matrix()
        
        return True
        
    except Exception as e:
        print(f"[Imprint] Error creating speaker imprint for {firstname}: {e}")
        return False

class OnlineECAPAProcessor:
    """
    Handles online ECAPA embedding extraction and speaker matching during live audio processing.
    Tracks timing and manages the extraction schedule.
    """
    
    def __init__(self, ecapa_embedder, ecapa_matcher, sample_rate=16000):
        """
        Initialize the online processor.
        
        Args:
            ecapa_embedder (ECAPASpeakerEmbedder): The ECAPA model wrapper
            ecapa_matcher (FastECAPASpeakerMatcher): The speaker matching system
            sample_rate (int): Audio sample rate (default: 16000)
        """
        self.ecapa_embedder = ecapa_embedder
        self.ecapa_matcher = ecapa_matcher
        self.sample_rate = sample_rate
        
        # Timing and extraction state
        self.bytes_per_second = sample_rate * 2  # 16-bit audio = 2 bytes per sample
        self.last_extraction_bytes = 0
        self.extraction_interval_bytes = self.bytes_per_second  # Extract every 1 second
        self.max_extractions = 3  # Stop after 3 seconds
        self.extraction_count = 0
        
        # Thresholds for speaker identification
        self.MAYBE_THRESHOLD = 0.70
        self.DEFINITELY_THRESHOLD = 0.85
    
    def reset_for_new_utterance(self):
        """Reset the processor state for a new utterance."""
        self.last_extraction_bytes = 0
        self.extraction_count = 0
        print("[OnlineECAPA] Reset for new utterance")
    
    def should_extract_now(self, buffer_size_bytes):
        """
        Determine if we should extract an embedding based on buffer size.
        
        Args:
            buffer_size_bytes (int): Current size of the audio buffer in bytes
            
        Returns:
            bool: True if we should extract an embedding now
        """
        if self.extraction_count >= self.max_extractions:
            return False
        
        bytes_since_last = buffer_size_bytes - self.last_extraction_bytes
        return bytes_since_last >= self.extraction_interval_bytes
    
    async def extract_and_match(self, audio_buffer, reason="scheduled"):
        """
        Extract ECAPA embedding from audio buffer and find best speaker match.
        
        Args:
            audio_buffer (bytes): Audio buffer as int16 PCM bytes
            reason (str): Reason for extraction ("scheduled", "silence", etc.)
            
        Returns:
            dict: Results containing speaker match information
        """
        try:
            buffer_duration = len(audio_buffer) / self.bytes_per_second
            print(f"[OnlineECAPA] Extracting embedding ({reason}) - buffer duration: {buffer_duration:.2f}s")
            
            # Convert bytes to numpy array
            audio_int16 = np.frombuffer(audio_buffer, dtype=np.int16)
            
            # Extract ECAPA embedding in thread pool to avoid blocking
            ecapa_embedding = await asyncio.to_thread(
                self.ecapa_embedder.extract_embedding_from_buffer, 
                audio_int16, 
                self.sample_rate
            )
            
            if ecapa_embedding is None:
                return {"error": "Failed to extract embedding"}
            
            # Find best speaker match
            speaker_name, uid, confidence = self.ecapa_matcher.find_best_match(ecapa_embedding)
            
            # Determine speaker identification result
            if confidence < self.MAYBE_THRESHOLD:
                speaker_result = "unknown speaker"
            elif confidence < self.DEFINITELY_THRESHOLD:
                speaker_result = f"{speaker_name}(?)"
            else:
                speaker_result = f"{speaker_name}"
            
            result = {
                "speaker_name": speaker_name,
                "uid": uid,
                "confidence": confidence,
                "speaker_result": speaker_result,
                "buffer_duration": buffer_duration,
                "extraction_reason": reason,
                "extraction_count": self.extraction_count + 1
            }
            
            print(f"[OnlineECAPA] Speaker match result: {speaker_result} (confidence: {confidence:.3f})")
            
            # Update extraction tracking if this was a scheduled extraction
            if reason == "scheduled":
                self.extraction_count += 1
                self.last_extraction_bytes = len(audio_buffer)
            
            return result
            
        except Exception as e:
            print(f"[OnlineECAPA] Error in extract_and_match: {e}")
            return {"error": str(e)}

# Add this to your global variables section (near the top of the file)
ecapa_embedder = None
online_ecapa_processor = None

# Add this to your main() function after initializing other models:
async def main():
    global main_loop, nemo_transcriber, pipertts_wrapper, xtts_wrapper, nemo_vad, canary_qwen_transcriber, ecapa_matcher
    global ecapa_embedder, online_ecapa_processor  # Add these globals
    
    main_loop = asyncio.get_event_loop()
    setup_database()
    
    # ... your existing model initializations ...
    
    # Initialize ECAPA speaker embedder (add this after other model inits)
    ecapa_embedder = ECAPASpeakerEmbedder(
        model_path=CONFIG["ecapa_tdnn_model_path"],
        device=CONFIG["inference_device"]
    )
    
    # ... your existing code continues ...
    
    # Load in-memory ECAPA matching routine
    ecapa_matcher = FastECAPASpeakerMatcher(con)
    
    # Initialize online ECAPA processor
    online_ecapa_processor = OnlineECAPAProcessor(
        ecapa_embedder=ecapa_embedder,
        ecapa_matcher=ecapa_matcher,
        sample_rate=CONFIG["audio_sample_rate"]
    )
    
    # ... rest of your main() function ...

# Modify your process_audio_from_queue function to integrate ECAPA processing
# Add this code to the voice activity processing section:

# In process_audio_from_queue(), add this after the VAD detection section:
"""
                    if is_voice_active_in_chunk:
                        current_utterance_buffer += chunk_bytes # Accumulate all speech
                        silence_counter = 0 # Reset silence counter

                        if not is_speaking:
                            is_speaking = True
                            # Reset ASR state for a new utterance
                            # ... your existing ASR reset code ...
                            
                            # NEW: Reset ECAPA processor for new utterance
                            online_ecapa_processor.reset_for_new_utterance()

                        # Perform ASR transcription on the *current audio chunk* if speech is active
                        text = await asyncio.to_thread(nemo_transcriber.transcribe_chunk, audio_chunk_np)
                        final_transcription_text = text

                        # NEW: Check if we should extract ECAPA embedding
                        if online_ecapa_processor.should_extract_now(len(current_utterance_buffer)):
                            ecapa_result = await online_ecapa_processor.extract_and_match(
                                current_utterance_buffer, 
                                reason="scheduled"
                            )
                            # You can log or process the ecapa_result as needed
                            if "error" not in ecapa_result:
                                print(f"[Speaker ID] {ecapa_result['speaker_result']}")

                        # ... rest of your existing voice processing code ...

                    else: # VAD indicates silence
                        silence_counter += 1
                        if is_speaking and silence_counter >= SILENCE_CHUNKS_THRESHOLD:
                            print("Acoustic finality detected. Processing full utterance...")
                            
                            # NEW: Extract final ECAPA embedding before clearing buffer
                            if len(current_utterance_buffer) > 0:
                                final_ecapa_result = await online_ecapa_processor.extract_and_match(
                                    current_utterance_buffer,
                                    reason="silence"
                                )
                                if "error" not in final_ecapa_result:
                                    print(f"[Final Speaker ID] {final_ecapa_result['speaker_result']}")
                            
                            # ... your existing finality processing code ...
                            
                            # Reset the buffer and state for the next utterance
                            is_speaking = False
                            silence_counter = 0
                            current_utterance_buffer = b''
                            # NEW: Reset ECAPA processor
                            online_ecapa_processor.reset_for_new_utterance()
"""