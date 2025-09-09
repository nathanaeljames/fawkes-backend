import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import asyncio
from typing import Tuple, Optional, List, Dict

class ECAPA_SpeakerMatcher:
    """
    In-memory speaker matcher for ECAPA-TDNN embeddings with adaptive confidence scoring.
    Loads all speaker embeddings once at startup for fast comparisons.
    """
    
    def __init__(self, duckdb_connection):
        """
        Initialize the matcher and load all ECAPA embeddings into memory.
        
        Args:
            duckdb_connection: Active DuckDB connection object
        """
        self.con = duckdb_connection
        self.speaker_embeddings = {}  # speaker_name -> numpy array
        self.embedding_matrix = None  # Combined matrix for vectorized operations
        self.speaker_names = []       # Ordered list matching matrix rows
        self.load_embeddings_to_memory()
    
    def load_embeddings_to_memory(self):
        """
        Load all ECAPA embeddings from database into memory for fast comparison.
        Called once during initialization.
        """
        print("Loading ECAPA embeddings into memory...")
        
        try:
            # Query all speakers with ECAPA embeddings
            speakers_query = self.con.execute("""
                SELECT firstname, surname, ecapa_embedding 
                FROM speakers 
                WHERE ecapa_embedding IS NOT NULL
            """).fetchall()
            
            if not speakers_query:
                print("No ECAPA embeddings found in database.")
                return
            
            embeddings_list = []
            
            for row in speakers_query:
                firstname, surname, ecapa_embedding_list = row
                
                # Reconstruct speaker name
                speaker_name = f"{firstname}_{surname}" if surname else firstname
                
                try:
                    # Convert DuckDB array (Python list) to numpy array
                    embedding_array = np.array(ecapa_embedding_list, dtype=np.float32)
                    
                    # Normalize the embedding for cosine similarity
                    embedding_normalized = embedding_array / np.linalg.norm(embedding_array)
                    
                    # Store in dictionary
                    self.speaker_embeddings[speaker_name] = embedding_normalized
                    
                    # Add to list for matrix construction
                    embeddings_list.append(embedding_normalized)
                    self.speaker_names.append(speaker_name)
                    
                    print(f"Loaded ECAPA embedding for {speaker_name} (shape: {embedding_array.shape})")
                    
                except Exception as e:
                    print(f"Error loading ECAPA embedding for {speaker_name}: {e}")
                    continue
            
            if embeddings_list:
                # Create combined matrix for vectorized operations
                self.embedding_matrix = np.vstack(embeddings_list)
                print(f"Created embedding matrix: {self.embedding_matrix.shape}")
                print(f"Loaded {len(self.speaker_embeddings)} ECAPA embeddings into memory.")
            else:
                print("No valid ECAPA embeddings loaded.")
                
        except Exception as e:
            print(f"Error loading ECAPA embeddings: {e}")
    
    def calculate_adaptive_confidence(self, best_score: float, second_score: float, domain_size: int) -> float:
        """
        Calculate confidence score that adapts based on domain size.
        
        Args:
            best_score: Highest cosine similarity score
            second_score: Second highest cosine similarity score
            domain_size: Number of speakers in consideration domain
            
        Returns:
            Composite confidence score (0.0 to 1.0)
        """
        base_confidence = best_score
        gap = best_score - second_score
        
        # Dynamic gap weighting: inversely proportional to domain size
        # Tunable parameters:
        # - 12.0: Controls gap sensitivity (higher = more sensitive for small domains)
        # - 0.4: Maximum gap weight ceiling
        gap_weight = min(0.4, 12.0 / domain_size)
        
        gap_bonus = gap * gap_weight
        composite = min(1.0, base_confidence + gap_bonus)
        
        return composite
    
    def find_best_match(self, query_embedding: np.ndarray, domain_size: Optional[int] = None) -> Tuple[Optional[str], float]:
        """
        Find the best matching speaker with adaptive confidence scoring.
        
        Args:
            query_embedding: ECAPA embedding to match (numpy array)
            domain_size: Number of speakers in consideration domain 
                        (defaults to total database size if not specified)
            
        Returns:
            Tuple of (speaker_name, confidence_score) or (None, 0.0) if no embeddings loaded
        """
        if self.embedding_matrix is None or len(self.speaker_embeddings) == 0:
            return None, 0.0
        
        # Use total database size if domain_size not specified
        if domain_size is None:
            domain_size = len(self.speaker_embeddings)
        
        try:
            # Normalize the query embedding
            query_normalized = query_embedding / np.linalg.norm(query_embedding)
            query_normalized = query_normalized.reshape(1, -1)
            
            # Compute cosine similarities with all speakers at once (vectorized)
            similarities = cosine_similarity(query_normalized, self.embedding_matrix)[0]
            
            # Get top 2 scores for confidence calculation
            sorted_indices = np.argsort(similarities)[::-1]  # Sort descending
            best_score = similarities[sorted_indices[0]]
            second_score = similarities[sorted_indices[1]] if len(similarities) > 1 else 0.0
            
            # Get best matching speaker
            best_speaker = self.speaker_names[sorted_indices[0]]
            
            # Calculate adaptive confidence
            confidence = self.calculate_adaptive_confidence(best_score, second_score, domain_size)
            
            return best_speaker, confidence
            
        except Exception as e:
            print(f"Error in speaker matching: {e}")
            return None, 0.0
    
    def find_best_match_with_details(self, query_embedding: np.ndarray, domain_size: Optional[int] = None) -> Dict:
        """
        Find best match with detailed breakdown for debugging/analysis.
        
        Returns:
            Dictionary with detailed matching information
        """
        if self.embedding_matrix is None or len(self.speaker_embeddings) == 0:
            return {"speaker": None, "confidence": 0.0, "details": "No embeddings loaded"}
        
        if domain_size is None:
            domain_size = len(self.speaker_embeddings)
        
        try:
            query_normalized = query_embedding / np.linalg.norm(query_embedding)
            query_normalized = query_normalized.reshape(1, -1)
            
            similarities = cosine_similarity(query_normalized, self.embedding_matrix)[0]
            sorted_indices = np.argsort(similarities)[::-1]
            
            best_score = similarities[sorted_indices[0]]
            second_score = similarities[sorted_indices[1]] if len(similarities) > 1 else 0.0
            best_speaker = self.speaker_names[sorted_indices[0]]
            
            gap = best_score - second_score
            gap_weight = min(0.4, 12.0 / domain_size)
            gap_bonus = gap * gap_weight
            confidence = min(1.0, best_score + gap_bonus)
            
            return {
                "speaker": best_speaker,
                "confidence": confidence,
                "details": {
                    "raw_similarity": best_score,
                    "second_best_similarity": second_score,
                    "similarity_gap": gap,
                    "gap_weight": gap_weight,
                    "gap_bonus": gap_bonus,
                    "domain_size": domain_size,
                    "total_speakers_loaded": len(self.speaker_embeddings)
                }
            }
            
        except Exception as e:
            return {"speaker": None, "confidence": 0.0, "details": f"Error: {e}"}
    
    def get_speaker_count(self) -> int:
        """Get the total number of speakers loaded in memory."""
        return len(self.speaker_embeddings)
    
    def reload_embeddings(self):
        """Reload embeddings from database (useful after new speaker enrollment)."""
        print("Reloading ECAPA embeddings from database...")
        self.speaker_embeddings.clear()
        self.speaker_names.clear()
        self.embedding_matrix = None
        self.load_embeddings_to_memory()

# Integration example for your existing code:
"""
# Add this to your main() function after database setup:

# Initialize ECAPA speaker matcher (loads embeddings once)
global ecapa_matcher
ecapa_matcher = ECAPA_SpeakerMatcher(con)

# Usage in your audio processing pipeline:
def process_speaker_identification(audio_buffer):
    # Extract ECAPA embedding from audio
    ecapa_embedding = extract_ecapa_embedding(audio_buffer)  # Your existing function
    
    # Find best match with adaptive confidence
    speaker_name, confidence = ecapa_matcher.find_best_match(ecapa_embedding)
    
    # Apply your thresholds
    MAYBE_THRESHOLD = 0.70
    DEFINITELY_THRESHOLD = 0.85
    
    if confidence < MAYBE_THRESHOLD:
        return "unknown"
    elif confidence < DEFINITELY_THRESHOLD:
        return f"{speaker_name}(?)"  # Grey with question mark
    else:
        return speaker_name  # Black, confident

# For debugging/analysis:
def debug_speaker_match(audio_buffer):
    ecapa_embedding = extract_ecapa_embedding(audio_buffer)
    details = ecapa_matcher.find_best_match_with_details(ecapa_embedding)
    print(f"Match details: {details}")
"""