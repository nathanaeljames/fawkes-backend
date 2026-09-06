import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

class InMemorySpeakerDatabase:
    """
    High-performance in-memory speaker recognition system.
    Loads all ECAPA embeddings at startup for fast similarity search.
    """
    
    def __init__(self, device="cuda"):
        self.device = device
        self.speaker_embeddings = {}  # speaker_uid -> embedding tensor
        self.speaker_metadata = {}    # speaker_uid -> {firstname, surname, etc.}
        self.embedding_matrix = None  # Batched tensor for vectorized operations
        self.speaker_uids = []        # Ordered list of UIDs matching embedding_matrix rows
        
    def load_from_database(self):
        """
        Load all ECAPA embeddings from database into memory once at startup.
        """
        print("Loading ECAPA embeddings into memory...")
        
        # Query all speakers with ECAPA embeddings
        query = """
        SELECT uid, firstname, surname, ecapa_embedding 
        FROM speakers 
        WHERE ecapa_embedding IS NOT NULL
        """
        
        results = con.execute(query).fetchall()
        
        embeddings_list = []
        
        for uid, firstname, surname, ecapa_array in results:
            # Convert native array back to tensor
            embedding_tensor = torch.tensor(ecapa_array, device=self.device, dtype=torch.float32)
            
            # Normalize embedding for cosine similarity (do once at load time)
            embedding_tensor = F.normalize(embedding_tensor, p=2, dim=0)
            
            self.speaker_embeddings[uid] = embedding_tensor
            self.speaker_metadata[uid] = {
                'firstname': firstname,
                'surname': surname,
                'full_name': f"{firstname} {surname}" if surname else firstname
            }
            
            embeddings_list.append(embedding_tensor)
            self.speaker_uids.append(uid)
        
        # Create batched tensor for vectorized similarity computation
        if embeddings_list:
            self.embedding_matrix = torch.stack(embeddings_list)
            print(f"Loaded {len(embeddings_list)} speaker embeddings into memory")
            print(f"Embedding matrix shape: {self.embedding_matrix.shape}")
        else:
            print("No ECAPA embeddings found in database")
    
    def identify_speaker_fast(self, query_embedding: torch.Tensor, threshold: float = 0.7) -> Optional[Tuple[int, str, float]]:
        """
        Ultra-fast speaker identification using vectorized operations.
        
        Args:
            query_embedding: ECAPA embedding tensor for unknown speaker
            threshold: Minimum similarity threshold
            
        Returns:
            Tuple of (speaker_uid, full_name, similarity_score) or None if no match
        """
        if self.embedding_matrix is None or len(self.speaker_uids) == 0:
            return None
        
        # Normalize query embedding
        query_normalized = F.normalize(query_embedding, p=2, dim=0)
        
        # Vectorized cosine similarity computation (ALL speakers at once)
        similarities = torch.mv(self.embedding_matrix, query_normalized)
        
        # Find best match
        best_idx = torch.argmax(similarities)
        best_similarity = similarities[best_idx].item()
        
        if best_similarity >= threshold:
            best_uid = self.speaker_uids[best_idx]
            speaker_info = self.speaker_metadata[best_uid]
            return (best_uid, speaker_info['full_name'], best_similarity)
        
        return None
    
    def identify_speaker_top_k(self, query_embedding: torch.Tensor, k: int = 5, threshold: float = 0.5) -> List[Tuple[int, str, float]]:
        """
        Get top-k most similar speakers.
        """
        if self.embedding_matrix is None or len(self.speaker_uids) == 0:
            return []
        
        query_normalized = F.normalize(query_embedding, p=2, dim=0)
        similarities = torch.mv(self.embedding_matrix, query_normalized)
        
        # Get top-k indices and scores
        top_k_values, top_k_indices = torch.topk(similarities, min(k, len(similarities)))
        
        results = []
        for i in range(len(top_k_values)):
            similarity = top_k_values[i].item()
            if similarity >= threshold:
                uid = self.speaker_uids[top_k_indices[i]]
                speaker_info = self.speaker_metadata[uid]
                results.append((uid, speaker_info['full_name'], similarity))
        
        return results
    
    def add_speaker_embedding(self, uid: int, embedding: torch.Tensor, firstname: str, surname: str = None):
        """
        Add a new speaker embedding to the in-memory database.
        Call this when you add a new speaker to rebuild the embedding matrix.
        """
        embedding_normalized = F.normalize(embedding, p=2, dim=0)
        self.speaker_embeddings[uid] = embedding_normalized
        self.speaker_metadata[uid] = {
            'firstname': firstname,
            'surname': surname,
            'full_name': f"{firstname} {surname}" if surname else firstname
        }
        
        # Rebuild the embedding matrix
        self._rebuild_embedding_matrix()
    
    def _rebuild_embedding_matrix(self):
        """Rebuild the batched embedding matrix after adding/removing speakers."""
        embeddings_list = []
        self.speaker_uids = []
        
        for uid, embedding in self.speaker_embeddings.items():
            embeddings_list.append(embedding)
            self.speaker_uids.append(uid)
        
        if embeddings_list:
            self.embedding_matrix = torch.stack(embeddings_list)
            print(f"Rebuilt embedding matrix: {self.embedding_matrix.shape}")

# Usage in your main application:
speaker_db = None

async def initialize_speaker_recognition():
    """Initialize the in-memory speaker database at startup."""
    global speaker_db
    speaker_db = InMemorySpeakerDatabase(device=CONFIG["inference_device"])
    await asyncio.to_thread(speaker_db.load_from_database)

# Fast speaker identification during audio processing:
def identify_speaker_realtime(ecapa_embedding):
    """
    Real-time speaker identification with sub-millisecond latency.
    No disk I/O - pure in-memory tensor operations.
    """
    if speaker_db is None:
        return None
    
    result = speaker_db.identify_speaker_fast(ecapa_embedding, threshold=0.7)
    if result:
        uid, name, similarity = result
        print(f"Identified speaker: {name} (similarity: {similarity:.3f})")
        return result
    else:
        print("Unknown speaker")
        return None

# Add this to your main() function:
async def main():
    global main_loop, nemo_transcriber, pipertts_wrapper, xtts_wrapper, nemo_vad, canary_qwen_transcriber
    main_loop = asyncio.get_event_loop()
    
    # Setup database and models...
    setup_database()
    
    # Initialize in-memory speaker recognition
    await initialize_speaker_recognition()
    
    # Continue with the rest of your initialization...
    pipertts_wrapper = PiperTTS(CONFIG["piper_model_path"])
    # ... etc