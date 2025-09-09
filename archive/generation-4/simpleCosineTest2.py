# This script provides a simple, standalone function to test raw cosine similarity
# of speaker embeddings, helping to isolate issues from more complex confidence methods.

import torch
import duckdb
import numpy as np
from pathlib import Path
import nemo.collections.asr as nemo_asr
import wave
import sys

# --- Configuration (Modify as needed) ---
# Assuming the same DuckDB database and directory for audio files
DB_PATH = "./speakers/speakers.duckdb"
AUDIO_DIR = "/root/fawkes/audio_samples/_preprocessed"

# --- Replicating your server's embedding extraction logic ---

def extract_embedding_from_file_replicated(model, wav_path):
    """
    Replicates the exact embedding extraction logic from the server file to ensure
    the query embedding is created in the same way as the stored embeddings.
    """
    try:
        print(f"[REPLICATED] Extracting embedding from file: {wav_path}")

        # The model's get_embedding method is designed to handle file paths.
        with torch.no_grad():
            embeddings = model.get_embedding(wav_path)

        # Move to CPU and convert to numpy, just as your original code does
        embedding_np = embeddings.cpu().numpy().squeeze()

        print(f"[REPLICATED] Extracted embedding shape: {embedding_np.shape}")
        return embedding_np
    except Exception as e:
        print(f"[REPLICATED] Error extracting embedding from file {wav_path}: {e}")
        return None

def extract_embedding_from_buffer(model, audio_int16, sample_rate=None):
    """
    Extract ECAPA embedding from int16 PCM audio buffer using the model's forward method.
    Based on the working tensor approach from geminiECAPAfromTensor3.py
    
    This function is intentionally left in its original form to replicate the failure.
    """
    # Use the model's internal sample rate if not provided
    if sample_rate is None:
        sample_rate = model.cfg.sample_rate

    try:
        print(f"[ECAPA] Extracting embedding from buffer: shape={audio_int16.shape}, sample_rate={sample_rate}")

        # Convert int16 to float32 in range [-1, 1] (same as proof of concept)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio_float32)

        # Ensure audio_tensor is 1D, then add batch dimension
        if audio_tensor.dim() > 1:
            if audio_tensor.shape[0] > 1:
                print("[ECAPA] Warning: Input is multi-channel. Selecting the first channel.")
                audio_tensor = audio_tensor[0]
            else:
                audio_tensor = audio_tensor.squeeze()

        # Add batch dimension: [B, T] where B=1
        audio_waveform = audio_tensor.unsqueeze(0)

        # Move tensor to the correct device
        audio_waveform = audio_waveform.to(model.device)

        # Create length tensor
        audio_len = torch.tensor([audio_waveform.shape[1]], dtype=torch.long, device=model.device)

        print(f"[ECAPA] Prepared audio tensor with shape: {audio_waveform.shape}")

        # Use the model's forward method directly (same as proof of concept)
        with torch.no_grad():
            # The forward() method returns a tuple. The first element is the logits, second is the embedding.
            _, embedding = model.forward(input_signal=audio_waveform, input_signal_length=audio_len)

        embedding_np = embedding.cpu().numpy().squeeze()

        print(f"[ECAPA] Extracted embedding shape from buffer: {embedding_np.shape}")
        return embedding_np

    except Exception as e:
        print(f"[ECAPA] Error extracting embedding from buffer: {e}")
        return None

def extract_embedding_from_buffer_corrected(model, audio_int16):
    """
    This function is intentionally left in its original, incorrect form to replicate the failure
    and demonstrate the `NeuralTypeComparisonResult.INCOMPATIBLE` error.
    """
    try:
        print(f"[ECAPA-CORRECTED] Extracting embedding from buffer: shape={audio_int16.shape}")

        # Convert int16 to float32 and add batch dimension
        audio_float32 = torch.from_numpy(audio_int16).unsqueeze(0).float()

        # The model's preprocessor expects an audio signal and a length tensor.
        audio_len = torch.tensor([audio_float32.shape[1]], dtype=torch.long, device=model.device)
        audio_float32 = audio_float32.to(model.device)

        # Use the preprocessor to convert the waveform to a feature tensor
        processed_audio, processed_audio_len = model.preprocessor(
            input_signal=audio_float32, length=audio_len
        )

        print(f"[ECAPA-CORRECTED] Prepared feature tensor with shape: {processed_audio.shape}")

        # Now pass the feature tensor to the model's forward method
        with torch.no_grad():
            _, embedding = model.forward(input_signal=processed_audio, input_signal_length=processed_audio_len)

        embedding_np = embedding.cpu().numpy().squeeze()

        print(f"[ECAPA-CORRECTED] Extracted embedding shape from buffer: {embedding_np.shape}")
        return embedding_np

    except Exception as e:
        print(f"[ECAPA-CORRECTED] Error extracting embedding from buffer: {e}")
        return None
        
def extract_embedding_from_buffer_final(model, audio_int16):
    """
    FINAL CORRECTED VERSION: This function extracts the ECAPA embedding from a
    raw audio buffer by formatting the tensor correctly and passing it directly
    to the model's forward method, which handles the internal pre-processing.
    
    The previous 'corrected' attempt failed because the forward() method of this
    specific model does not accept pre-processed features (like MelSpectrograms)
    as input. It is designed to take the raw waveform.
    """
    try:
        print(f"[ECAPA-FINAL] Extracting embedding from buffer: shape={audio_int16.shape}")

        # Convert int16 to float32 and normalize to [-1, 1]
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        
        # Convert to torch tensor with a batch dimension of 1.
        # This matches the (batch, time) shape the forward method expects.
        audio_tensor = torch.from_numpy(audio_float32).unsqueeze(0).to(model.device)

        # Create length tensor
        audio_len = torch.tensor([audio_tensor.shape[1]], dtype=torch.long, device=model.device)
        
        print(f"[ECAPA-FINAL] Prepared audio tensor with shape: {audio_tensor.shape}")

        with torch.no_grad():
            # Pass the raw audio signal directly to the forward method
            _, embedding = model.forward(input_signal=audio_tensor, input_signal_length=audio_len)

        embedding_np = embedding.cpu().numpy().squeeze()
        
        print(f"[ECAPA-FINAL] Extracted embedding shape from buffer: {embedding_np.shape}")
        return embedding_np

    except Exception as e:
        print(f"[ECAPA-FINAL] Error extracting embedding from buffer: {e}")
        return None

# --- Function for Simple Cosine Similarity ---

def calculate_cosine_similarity_simple(query_embedding: torch.Tensor, speaker_embeddings: dict) -> tuple:
    """
    Calculates raw cosine similarity between a single query embedding and a
    dictionary of speaker embeddings, returning the best match and score.

    Args:
        query_embedding (torch.Tensor): The embedding of the unknown speaker.
        speaker_embeddings (dict): A dictionary where keys are speaker IDs and
                                   values are the corresponding embeddings.

    Returns:
        tuple: A tuple containing the best matching speaker's ID and the
               raw cosine similarity score.
    """
    best_match_id = "unknown speaker"
    max_similarity = -1.0  # Cosine similarity ranges from -1 to 1

    # Ensure the query embedding is normalized (this is good practice)
    # The to('cpu') call ensures that the query embedding is on the CPU for comparison
    query_embedding = query_embedding.to('cpu')
    query_embedding = query_embedding / torch.linalg.norm(query_embedding, ord=2)

    # Iterate through all speaker embeddings in the database
    for speaker_id, speaker_emb in speaker_embeddings.items():
        # Ensure the speaker embedding is also normalized and on the same device as the query embedding
        speaker_emb = speaker_emb.to(query_embedding.device)
        speaker_emb = speaker_emb / torch.linalg.norm(speaker_emb, ord=2)
        
        # Calculate cosine similarity using dot product
        # The squeeze() operation is important here to ensure the dot product works correctly
        similarity = torch.dot(query_embedding.squeeze(), speaker_emb.squeeze()).item()

        # Update the best match if the current similarity is higher
        if similarity > max_similarity:
            max_similarity = similarity
            best_match_id = speaker_id
            
    return best_match_id, max_similarity

# --- Main Test Script ---

if __name__ == "__main__":
    print("--- Starting simple cosine similarity test ---")

    try:
        # Initialize the ECAPA-TDNN speaker embedding model from the offline path
        OFFLINE_MODEL_PATH = "/root/fawkes/models/ecapa_tdnn_embed/ecapa_tdnn.nemo"
        if not Path(OFFLINE_MODEL_PATH).exists():
            print(f"Error: Offline model not found at '{OFFLINE_MODEL_PATH}'. Please check the path.")
            sys.exit(1)
            
        ecapa_processor = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(restore_path=OFFLINE_MODEL_PATH)
        
        # Connect to the DuckDB database
        con = duckdb.connect(DB_PATH)
        
        # 1. Load the speaker embeddings from the database for comparison
        print(f"Loading speaker embeddings from '{DB_PATH}'...")
        speaker_embeddings_from_db = {}
        speaker_details = {}  # Dictionary to store firstname and surname
        
        results = con.execute("SELECT uid, ecapa_embedding, firstname, surname FROM speakers").fetchall()
        
        if not results:
            print("No speakers found in the database. Please run the main server script to add speakers first.")
            sys.exit(1)
            
        for row in results:
            uid, embedding, firstname, surname = row
            # The embedding is a native array, so we can directly convert it to a PyTorch tensor
            speaker_embeddings_from_db[uid] = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
            speaker_details[uid] = f"{firstname}_{surname}"
            
        print(f"Successfully loaded {len(speaker_embeddings_from_db)} speaker embeddings.")

        # 2. Test using the file-based extraction method (your first working method)
        test_audio_path = Path(AUDIO_DIR) / "nathanael_01.wav"
        if not test_audio_path.exists():
            print(f"Error: Test audio file not found at '{test_audio_path}'. Please check the path and filename.")
            sys.exit(1)

        print(f"\n--- Testing extract_embedding_from_file_replicated() ---")
        query_embedding_np_file = extract_embedding_from_file_replicated(ecapa_processor, str(test_audio_path))
        if query_embedding_np_file is None:
            print("Failed to extract query embedding from file.")
        else:
            query_embedding_file = torch.from_numpy(query_embedding_np_file).unsqueeze(0)
            best_match_uid_file, similarity_score_file = calculate_cosine_similarity_simple(
                query_embedding_file, 
                speaker_embeddings_from_db
            )
            print("\n--- Results from file-based test ---")
            best_match_name_file = speaker_details.get(best_match_uid_file, "unknown_speaker")
            print(f"Best matching speaker: {best_match_name_file} (UID: {best_match_uid_file})")
            print(f"Raw Cosine Similarity Score: {similarity_score_file:.4f}")

        # 3. Test using the buffer-based extraction method (your second method)
        print(f"\n--- Testing extract_embedding_from_buffer() --- (Expected to Fail)")
        try:
            with wave.open(str(test_audio_path), 'rb') as wf:
                if wf.getsampwidth() != 2: # 16-bit
                    raise ValueError("Audio file must be 16-bit PCM.")
                n_frames = wf.getnframes()
                audio_int16 = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)
        except Exception as e:
            print(f"Error loading audio buffer: {e}")
            sys.exit(1)

        print(f"Loaded audio buffer with shape: {audio_int16.shape}")
        
        # Pass the buffer directly to your original function (will fail)
        query_embedding_np_buffer_failed = extract_embedding_from_buffer(ecapa_processor, audio_int16)
        
        if query_embedding_np_buffer_failed is not None:
            print("Unexpected success from failing function.")
        else:
            print("Failed to extract embedding from original buffer function as expected.")
        
        # 4. Test using the corrected buffer-based extraction method
        print(f"\n--- Testing extract_embedding_from_buffer_corrected() ---")
        query_embedding_np_buffer_corrected = extract_embedding_from_buffer_corrected(ecapa_processor, audio_int16)

        if query_embedding_np_buffer_corrected is not None:
            print("Unexpected success from second failing function.")
        else:
            print("Failed to extract embedding from corrected buffer function as expected.")
            
        # 5. Test using the new, final corrected buffer-based extraction method
        print(f"\n--- Testing extract_embedding_from_buffer_final() ---")
        query_embedding_np_buffer_final = extract_embedding_from_buffer_final(ecapa_processor, audio_int16)

        if query_embedding_np_buffer_final is not None:
            query_embedding_buffer_final = torch.from_numpy(query_embedding_np_buffer_final).unsqueeze(0)
            best_match_uid_buffer_final, similarity_score_buffer_final = calculate_cosine_similarity_simple(
                query_embedding_buffer_final,
                speaker_embeddings_from_db
            )
            
            best_match_name_buffer_final = speaker_details.get(best_match_uid_buffer_final, "unknown_speaker")
            
            print("\n--- Results from final corrected buffer-based test ---")
            print(f"Best matching speaker: {best_match_name_buffer_final} (UID: {best_match_uid_buffer_final})")
            print(f"Raw Cosine Similarity Score: {similarity_score_buffer_final:.4f}")
        else:
            print("Failed to extract embedding from final corrected buffer function.")

        # Close the database connection
        con.close()

    except FileNotFoundError:
        print(f"Error: One of the required files was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
