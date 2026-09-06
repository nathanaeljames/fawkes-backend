# This script provides a simple, standalone function to test raw cosine similarity
# of speaker embeddings, helping to isolate issues from more complex confidence methods.

import torch
import duckdb
import numpy as np
from pathlib import Path
import nemo.collections.asr as nemo_asr

# --- Configuration (Modify as needed) ---
# Assuming the same DuckDB database and directory for audio files
# Updated to use the correct path provided in the query
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
        # This will load the model from the .nemo file you used for initial extraction
        OFFLINE_MODEL_PATH = "/root/fawkes/models/ecapa_tdnn_embed/ecapa_tdnn.nemo"
        if not Path(OFFLINE_MODEL_PATH).exists():
            print(f"Error: Offline model not found at '{OFFLINE_MODEL_PATH}'. Please check the path.")
            exit()
            
        ecapa_processor = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(restore_path=OFFLINE_MODEL_PATH)
        
        # Connect to the DuckDB database
        con = duckdb.connect(DB_PATH)
        
        # 1. Load the speaker embeddings from the database for comparison
        print(f"Loading speaker embeddings from '{DB_PATH}'...")
        speaker_embeddings_from_db = {}
        speaker_details = {}  # Dictionary to store firstname and surname
        
        # Fetch the embeddings from the 'speakers' table. The column name for the ID is 'uid' and the embedding is 'ecapa_embedding'
        results = con.execute("SELECT uid, ecapa_embedding, firstname, surname FROM speakers").fetchall()
        
        if not results:
            print("No speakers found in the database. Please run the main server script to add speakers first.")
            exit()
            
        for row in results:
            uid, embedding, firstname, surname = row
            # The embedding is a native array, so we can directly convert it to a PyTorch tensor
            speaker_embeddings_from_db[uid] = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
            speaker_details[uid] = f"{firstname}_{surname}"
            
        print(f"Successfully loaded {len(speaker_embeddings_from_db)} speaker embeddings.")

        # 2. Extract an embedding from a sample audio file for testing
        test_audio_path = Path(AUDIO_DIR) / "nathanael_01.wav"
        if not test_audio_path.exists():
            print(f"Error: Test audio file not found at '{test_audio_path}'. Please check the path and filename.")
            exit()

        print(f"\nExtracting embedding from '{test_audio_path}'...")
        
        # Use the new replicated function to create the query embedding
        query_embedding_np = extract_embedding_from_file_replicated(ecapa_processor, str(test_audio_path))
        if query_embedding_np is None:
            print("Failed to extract query embedding.")
            exit()
            
        # Convert the numpy array back to a PyTorch tensor for the comparison function
        query_embedding = torch.from_numpy(query_embedding_np).unsqueeze(0)
        
        print("Embedding extraction successful.")
        print(f"Shape of the extracted query embedding: {query_embedding.shape}")

        # 3. Perform the simple cosine similarity comparison
        print("\nPerforming simple cosine similarity calculation...")
        best_match_uid, similarity_score = calculate_cosine_similarity_simple(
            query_embedding, 
            speaker_embeddings_from_db
        )

        # 4. Print the results
        print("\n--- Results ---")
        best_match_name = speaker_details.get(best_match_uid, "unknown_speaker")
        print(f"Best matching speaker: {best_match_name} (UID: {best_match_uid})")
        print(f"Raw Cosine Similarity Score: {similarity_score:.4f}")
        
        # Close the database connection
        con.close()

    except FileNotFoundError:
        print(f"Error: One of the required files was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
