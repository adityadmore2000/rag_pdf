import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid
from typing import List, Dict, Any


class VectorStore:
    def __init__(self, embedding_manager, collection_name: str = "pdf_documents"):
        """
        Initialize Qdrant with automatic dimension detection.
        Args:
            embedding_manager: The instance of your EmbeddingManager class
        """
        self.collection_name = collection_name
        self.embedding_manager = embedding_manager
        
        # Connect to Qdrant (assumes Qdrant is in Docker, and this code is too)
        self.client = QdrantClient(host="localhost", port=6333)
        
        # --- THE AUTO-DETECT MAGIC ---
        # We grab one sample embedding to see how big it is
        print("Detecting vector dimensions from embedding model...")
        sample_vec = self.embedding_manager.generate_embeddings(["test"])[0]
        self.vector_size = len(sample_vec)
        print(f"Model detected: {self.vector_size} dimensions.")
        
        self._initialize_store()

    def _initialize_store(self):
        try:
            # Check if collection exists
            exists = self.client.collection_exists(collection_name=self.collection_name)
            
            if exists:
                # IMPORTANT: Check if the existing collection matches our current model
                info = self.client.get_collection(self.collection_name)
                existing_size = info.config.params.vectors.size
                
                if existing_size != self.vector_size:
                    print(f"⚠️ Warning: Collection has {existing_size} dims, but model provides {self.vector_size}.")
                    print("Recreating collection to match new model...")
                    self.client.delete_collection(self.collection_name)
                    exists = False  # Force recreation below

            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                )
                print(f"✅ Collection {self.collection_name} ready.")
                
        except Exception as e:
            print(f"❌ Error initializing Qdrant: {e}")
            raise

    def add_documents(self, documents, embeddings):
        """Add documents with their embeddings to the vector store"""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        points = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Qdrant uses PointStruct to wrap ID, Vector, and Payload (metadata + text)
            points.append(PointStruct(
                id=str(uuid.uuid4()),  # Qdrant IDs must be UUIDs or integers
                vector=embedding.tolist(),
                payload={
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "doc_index": i
                }
            ))
        
        try:
            # Use self.client.upsert, NOT self.collection.upsert
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            count_result = self.client.count(collection_name=self.collection_name)
            print(f"Total documents now in collection: {count_result.count}")
            
        except Exception as e:
            print(f"Error adding to Qdrant: {e}")
            raise


# Example usage
if __name__ == "__main__":
    from embedding import EmbeddingManager
    from langchain_core.documents import Document
    
    # Initialize embedding manager
    embedding_manager = EmbeddingManager()
    
    # Initialize vector store
    vector_store = VectorStore(embedding_manager, collection_name="pdf_documents")
    
    # Create sample documents
    sample_docs = [
        Document(page_content="Machine learning is a subset of AI", metadata={"source_file": "doc1.pdf", "page": 1}),
        Document(page_content="Deep learning uses neural networks", metadata={"source_file": "doc2.pdf", "page": 1}),
    ]
    
    # Generate embeddings
    texts = [doc.page_content for doc in sample_docs]
    embeddings = embedding_manager.generate_embeddings(texts)
    
    # Add documents to vector store
    vector_store.add_documents(sample_docs, embeddings)
    print(f"✅ Added {len(sample_docs)} documents to vector store")