import numpy as np
from sentence_transformers import SentenceTransformer
import torch


class EmbeddingManager:
    """Evolved Librarian: Uses Nomic for long-context research analysis."""
    
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        self.model_name = model_name
        # Auto-detect GPU (cuda/mps) or fallback to CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        try:
            print(f"Loading {self.model_name} on {self.device}...")
            # trust_remote_code is required for some newer 2026 architectures
            self.model = SentenceTransformer(self.model_name, trust_remote_code=True)
            self.model.to(self.device) 
            
            self.dim = self.model.get_sentence_embedding_dimension()
            print(f"✅ Success! Dimensions: {self.dim}")
        except Exception as e:
            print(f"❌ Load Error: {e}")
            raise

    def generate_embeddings(self, texts: list) -> np.ndarray:
        """Generate embeddings for a list of texts
        
        Nomic requires a prefix for better retrieval accuracy:
        'search_document:' for indexing, 'search_query:' for the user's question
        """
        prefixed_texts = [f"search_document: {t}" for t in texts]
        
        return self.model.encode(
            prefixed_texts, 
            convert_to_numpy=True, 
            show_progress_bar=False
        )


# Example usage
if __name__ == "__main__":
    # Initialize embedding manager
    embedding_manager = EmbeddingManager()
    
    # Generate embeddings for sample texts
    sample_texts = [
        "This is a sample document about machine learning.",
        "Another document discussing artificial intelligence."
    ]
    
    embeddings = embedding_manager.generate_embeddings(sample_texts)
    print(f"Generated embeddings shape: {embeddings.shape}")
    print(f"Embedding dimensions: {embedding_manager.dim}")