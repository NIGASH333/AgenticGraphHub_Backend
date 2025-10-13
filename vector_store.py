"""
Vector Store for the RAG system

Handles FAISS vector storage and retrieval. 
TODO: Maybe add support for other vector stores later (Pinecone, Weaviate?)
"""

import os
import json
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import faiss
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# Set up logging - keeping it simple for now
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """Handles FAISS vector storage and retrieval operations."""
    
    def __init__(self, openai_api_key: str, index_dir: str = "./data/faiss_index"):
        """
        Initialize the VectorStore.
        
        Args:
            openai_api_key: OpenAI API key for embeddings
            index_dir: Directory to store FAISS index files
        """
        self.openai_api_key = openai_api_key
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings - using the cheaper model for now
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model="text-embedding-3-small"  # Cheaper than large model
        )
        
        # Text splitter - these numbers seem to work okay
        self.text_splitter = CharacterTextSplitter(
            chunk_size=1000,  # Could probably tune this better
            chunk_overlap=200,  # Some overlap to avoid losing context
            separator="\n"
        )
        
        # FAISS index and metadata
        self.index = None
        self.metadata = []
        self.dimension = 1536  # text-embedding-3-small dimension
        
        # Try to load existing index
        self._load_index()
    
    def _load_index(self):
        """Load existing FAISS index and metadata."""
        index_path = self.index_dir / "faiss_index.bin"
        metadata_path = self.index_dir / "metadata.json"
        
        if index_path.exists() and metadata_path.exists():
            try:
                # Load FAISS index
                self.index = faiss.read_index(str(index_path))
                
                # Load metadata
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                
                logger.info(f"Loaded existing index with {len(self.metadata)} vectors")
            except Exception as e:
                logger.error(f"Failed to load existing index: {e}")
                # Fallback to creating new index
                self._create_new_index()
        else:
            # No existing index, create new one
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index."""
        # Using IndexFlatIP for cosine similarity - not the fastest but simple
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        logger.info("Created new FAISS index")
    
    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]] = None) -> int:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries for each document
            
        Returns:
            Number of chunks added
        """
        if metadatas is None:
            metadatas = [{}] * len(documents)
        
        chunks_added = 0
        
        for doc, metadata in zip(documents, metadatas):
            # Split document into chunks
            chunks = self.text_splitter.split_text(doc)
            
            for i, chunk in enumerate(chunks):
                try:
                    # Generate embedding - this is where the API calls happen
                    embedding = self.embeddings.embed_query(chunk)
                    embedding_array = np.array([embedding], dtype=np.float32)
                    
                    # Add to index
                    self.index.add(embedding_array)
                    
                    # Store metadata for later retrieval
                    chunk_metadata = {
                        "text": chunk,
                        "chunk_index": i,
                        "document_metadata": metadata,
                        "vector_id": len(self.metadata)  # Simple ID assignment
                    }
                    self.metadata.append(chunk_metadata)
                    chunks_added += 1
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {e}")
                    # Continue with other chunks even if one fails
        
        # Save everything to disk
        self._save_index()
        
        logger.info(f"Added {chunks_added} chunks to vector store")
        return chunks_added
    
    def add_documents_from_files(self, data_dir: str) -> int:
        """
        Add documents from files in a directory.
        
        Args:
            data_dir: Path to directory containing documents
            
        Returns:
            Number of chunks added
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            logger.error(f"Data directory not found: {data_dir}")
            return 0
        
        total_chunks = 0
        supported_extensions = ['.txt', '.md', '.py']
        
        for file_path in data_path.glob('*'):
            if file_path.suffix.lower() in supported_extensions:
                try:
                    # Read file content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Add document
                    metadata = {"source_file": file_path.name}
                    chunks_added = self.add_documents([content], [metadata])
                    total_chunks += chunks_added
                    
                    logger.info(f"Processed {file_path.name}: {chunks_added} chunks")
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_path.name}: {e}")
        
        return total_chunks
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results with metadata
        """
        if self.index is None or len(self.metadata) == 0:
            logger.warning("No documents in vector store")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            query_array = np.array([query_embedding], dtype=np.float32)
            
            # Search - limit k to avoid index errors
            search_k = min(k, len(self.metadata))
            scores, indices = self.index.search(query_array, search_k)
            
            # Prepare results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.metadata):  # Safety check
                    result = {
                        "text": self.metadata[idx]["text"],
                        "score": float(score),
                        "metadata": self.metadata[idx]
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []  # Return empty list on error
    
    def get_similar_chunks(self, text: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Find chunks similar to the given text.
        
        Args:
            text: Text to find similar chunks for
            k: Number of results to return
            
        Returns:
            List of similar chunks
        """
        return self.search(text, k)
    
    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        try:
            # Save FAISS index
            index_path = self.index_dir / "faiss_index.bin"
            faiss.write_index(self.index, str(index_path))
            
            # Save metadata
            metadata_path = self.index_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved index with {len(self.metadata)} vectors")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "total_vectors": len(self.metadata),
            "dimension": self.dimension,
            "index_type": "FAISS IndexFlatIP"
        }
    
    def clear(self):
        """Clear all vectors from the store."""
        self._create_new_index()
        logger.info("Vector store cleared")
