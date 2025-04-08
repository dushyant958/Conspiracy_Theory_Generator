# vector_database.py
import logging
import json
import time
import argparse
from pathlib import Path
import numpy as np
import os
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# Optional imports for different vector DB backends
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Install with: pip install faiss-cpu or faiss-gpu")

try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logging.warning("Pinecone not available. Install with: pip install pinecone-client")

try:
    import qdrant_client
    from qdrant_client.models import VectorParams, Distance
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logging.warning("Qdrant not available. Install with: pip install qdrant-client")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/vector_db.log"),
        logging.StreamHandler()
    ]
)

class VectorDatabase:
    """Base class for vector database implementations"""
    def __init__(self, dimension: int, index_name: str):
        self.dimension = dimension
        self.index_name = index_name
        self.metadata_store = {}  # In-memory store for document metadata
    
    def add_documents(self, document_ids: List[str], embeddings: List[List[float]], 
                     metadata: List[Dict[str, Any]]) -> bool:
        """Add documents to the vector database"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def delete(self, document_ids: List[str]) -> bool:
        """Delete documents from the vector database"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def save(self, path: str) -> bool:
        """Save the vector database to disk"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def load(self, path: str) -> bool:
        """Load the vector database from disk"""
        raise NotImplementedError("Subclasses must implement this method")

class FaissVectorDB(VectorDatabase):
    """FAISS implementation of vector database"""
    def __init__(self, dimension: int, index_name: str, metric: str = 'cosine'):
        super().__init__(dimension, index_name)
        
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is not available. Install with: pip install faiss-cpu or faiss-gpu")
        
        # Create FAISS index
        if metric == 'cosine':
            # Normalize vectors and use L2 for cosine similarity
            self.index = faiss.IndexFlatL2(dimension)
            self.normalize = True
        elif metric == 'l2':
            self.index = faiss.IndexFlatL2(dimension)
            self.normalize = False
        elif metric == 'inner_product':
            self.index = faiss.IndexFlatIP(dimension)
            self.normalize = False
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        # Store document IDs to map FAISS indices to doc IDs
        self.doc_ids = []
        
        logging.info(f"Initialized FAISS vector database with dimension {dimension}")
    
    def _normalize_vectors(self, vectors):
        """Normalize vectors for cosine similarity"""
        if not self.normalize:
            return vectors
            
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms
    
    def add_documents(self, document_ids: List[str], embeddings: List[List[float]], 
                     metadata: List[Dict[str, Any]]) -> bool:
        """Add documents to the FAISS index"""
        try:
            if not document_ids or not embeddings:
                logging.warning("No documents to add")
                return False
            
            # Convert embeddings to numpy array
            vectors = np.array(embeddings, dtype=np.float32)
            
            # Normalize if needed
            vectors = self._normalize_vectors(vectors)
            
            # Add to FAISS index
            self.index.add(vectors)
            
            # Store document IDs and metadata
            start_idx = len(self.doc_ids)
            for i, doc_id in enumerate(document_ids):
                idx = start_idx + i
                self.doc_ids.append(doc_id)
                self.metadata_store[doc_id] = metadata[i]
            
            logging.info(f"Added {len(document_ids)} documents to FAISS index")
            return True
            
        except Exception as e:
            logging.error(f"Error adding documents to FAISS: {str(e)}")
            return False
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents in FAISS"""
        try:
            # Convert query to numpy array
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Normalize if needed
            query_vector = self._normalize_vectors(query_vector)
            
            # Search
            distances, indices = self.index.search(query_vector, min(top_k, len(self.doc_ids)))
            
            # Prepare results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self.doc_ids):
                    continue  # Skip invalid indices
                    
                doc_id = self.doc_ids[idx]
                result = {
                    'id': doc_id,
                    'score': float(distances[0][i]),
                    'metadata': self.metadata_store.get(doc_id, {})
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logging.error(f"Error searching in FAISS: {str(e)}")
            return []
    
    def delete(self, document_ids: List[str]) -> bool:
        """Delete documents from FAISS (requires rebuilding the index)"""
        try:
            # Filter out document IDs and rebuild index
            vectors = []
            new_doc_ids = []
            new_metadata = {}
            
            for doc_id in self.doc_ids:
                if doc_id not in document_ids:
                    # Keep this document
                    new_doc_ids.append(doc_id)
                    new_metadata[doc_id] = self.metadata_store.get(doc_id, {})
            
            # We need to retrieve vectors from the original source to rebuild
            logging.warning("FAISS delete operation requires rebuilding the index from source data")
            return False
            
        except Exception as e:
            logging.error(f"Error deleting documents from FAISS: {str(e)}")
            return False
    
    def save(self, path: str) -> bool:
        """Save FAISS index and metadata to disk"""
        try:
            save_path = Path(path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            index_path = save_path / f"{self.index_name}.index"
            faiss.write_index(self.index, str(index_path))
            
            # Save document IDs and metadata
            metadata_path = save_path / f"{self.index_name}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'doc_ids': self.doc_ids,
                    'metadata': self.metadata_store,
                    'dimension': self.dimension,
                    'normalize': self.normalize
                }, f)
            
            logging.info(f"Saved FAISS index to {path}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving FAISS index: {str(e)}")
            return False
    
    def load(self, path: str) -> bool:
        """Load FAISS index and metadata from disk"""
        try:
            load_path = Path(path)
            
            # Load FAISS index
            index_path = load_path / f"{self.index_name}.index"
            self.index = faiss.read_index(str(index_path))
            
            # Load document IDs and metadata
            metadata_path = load_path / f"{self.index_name}_metadata.json"
            with open(metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.doc_ids = data['doc_ids']
                self.metadata_store = data['metadata']
                self.dimension = data['dimension']
                self.normalize = data.get('normalize', False)
            
            logging.info(f"Loaded FAISS index from {path} with {len(self.doc_ids)} documents")
            return True
            
        except Exception as e:
            logging.error(f"Error loading FAISS index: {str(e)}")
            return False

class PineconeVectorDB(VectorDatabase):
    """Pinecone implementation of vector database"""
    def __init__(self, dimension: int, index_name: str, api_key: str = None, environment: str = None):
        super().__init__(dimension, index_name)
        
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone is not available. Install with: pip install pinecone-client")
        
        # Initialize Pinecone
        api_key = api_key or os.getenv("PINECONE_API_KEY")
        environment = environment or os.getenv("PINECONE_ENVIRONMENT")
        
        if not api_key or not environment:
            raise ValueError("Pinecone API key and environment must be provided")
        
        pinecone.init(api_key=api_key, environment=environment)
        
        # Create index if it doesn't exist
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine"
            )
            logging.info(f"Created new Pinecone index: {index_name}")
        
        self.index = pinecone.Index(index_name)
        logging.info(f"Connected to Pinecone index: {index_name}")
    
    def add_documents(self, document_ids: List[str], embeddings: List[List[float]], 
                     metadata: List[Dict[str, Any]]) -> bool:
        """Add documents to Pinecone"""
        try:
            if not document_ids or not embeddings:
                logging.warning("No documents to add")
                return False
            
            # Prepare vectors for upsert
            vectors = []
            for i, (doc_id, embedding) in enumerate(zip(document_ids, embeddings)):
                vectors.append((doc_id, embedding, metadata[i]))
            
            # Upsert in batches to avoid rate limits
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i+batch_size]
                
                # Format for Pinecone
                pinecone_batch = [(v[0], v[1], v[2]) for v in batch]
                
                # Upsert to Pinecone
                self.index.upsert(vectors=pinecone_batch)
                
                # Store metadata locally as well
                for doc_id, _, meta in batch:
                    self.metadata_store[doc_id] = meta
                
                logging.info(f"Upserted batch {i//batch_size + 1} with {len(batch)} vectors to Pinecone")
            
            return True
            
        except Exception as e:
            logging.error(f"Error adding documents to Pinecone: {str(e)}")
            return False
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents in Pinecone"""
        try:
            # Query Pinecone
            query_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Format results
            results = []
            for match in query_results.matches:
                result = {
                    'id': match.id,
                    'score': match.score,
                    'metadata': match.metadata
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logging.error(f"Error searching in Pinecone: {str(e)}")
            return []
    
    def delete(self, document_ids: List[str]) -> bool:
        """Delete documents from Pinecone"""
        try:
            self.index.delete(ids=document_ids)
            
            # Remove from local metadata store
            for doc_id in document_ids:
                if doc_id in self.metadata_store:
                    del self.metadata_store[doc_id]
            
            logging.info(f"Deleted {len(document_ids)} documents from Pinecone")
            return True
            
        except Exception as e:
            logging.error(f"Error deleting documents from Pinecone: {str(e)}")
            return False
    
    def save(self, path: str) -> bool:
        """Save Pinecone metadata to disk (index is in the cloud)"""
        try:
            save_path = Path(path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save metadata
            metadata_path = save_path / f"{self.index_name}_pinecone_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': self.metadata_store,
                    'dimension': self.dimension,
                    'index_name': self.index_name
                }, f)
            
            logging.info(f"Saved Pinecone metadata to {path}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving Pinecone metadata: {str(e)}")
            return False
    
    def load(self, path: str) -> bool:
        """Load Pinecone metadata from disk"""
        try:
            load_path = Path(path)
            
            # Load metadata
            metadata_path = load_path / f"{self.index_name}_pinecone_metadata.json"
            with open(metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.metadata_store = data['metadata']
                self.dimension = data['dimension']
            
            logging.info(f"Loaded Pinecone metadata from {path}")
            return True
            
        except Exception as e:
            logging.error(f"Error loading Pinecone metadata: {str(e)}")
            return False

class QdrantVectorDB(VectorDatabase):
    """Qdrant implementation of vector database"""
    def __init__(self, dimension: int, index_name: str, url: str = None, api_key: str = None):
        super().__init__(dimension, index_name)
        
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant is not available. Install with: pip install qdrant-client")
        
        # Initialize Qdrant client
        url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = api_key or os.getenv("QDRANT_API_KEY")
        
        self.client = qdrant_client.QdrantClient(url=url, api_key=api_key)
        
        # Create collection if it doesn't exist
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if index_name not in collection_names:
                self.client.create_collection(
                    collection_name=index_name,
                    vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
                )
                logging.info(f"Created new Qdrant collection: {index_name}")
        except Exception as e:
            logging.error(f"Error checking/creating Qdrant collection: {str(e)}")
        
        logging.info(f"Connected to Qdrant collection: {index_name}")
    
    def add_documents(self, document_ids: List[str], embeddings: List[List[float]], 
                    metadata: List[Dict[str, Any]]) -> bool:
        """Add documents to Qdrant"""
        try:
            if not document_ids or not embeddings:
                logging.warning("No documents to add")
                return False
            
            # Prepare points for upsert
            points = []
            for i, (doc_id, embedding) in enumerate(zip(document_ids, embeddings)):
                point = {
                    "id": doc_id,
                    "vector": embedding,
                    "payload": metadata[i]
                }
                points.append(point)
            
            # Upsert in batches to avoid rate limits
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                
                # Upsert to Qdrant
                self.client.upsert(
                    collection_name=self.index_name,
                    points=batch
                )
                
                # Store metadata locally as well
                for point in batch:
                    self.metadata_store[point["id"]] = point["payload"]
                
                logging.info(f"Upserted batch {i//batch_size + 1} with {len(batch)} vectors to Qdrant")
            
            return True
            
        except Exception as e:
            logging.error(f"Error adding documents to Qdrant: {str(e)}")
            return False
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents in Qdrant"""
        try:
            # Query Qdrant
            search_results = self.client.search(
                collection_name=self.index_name,
                query_vector=query_embedding,
                limit=top_k
            )
            
            # Format results
            results = []
            for match in search_results:
                result = {
                    'id': match.id,
                    'score': match.score,
                    'metadata': match.payload
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logging.error(f"Error searching in Qdrant: {str(e)}")
            return []
    
    def delete(self, document_ids: List[str]) -> bool:
        """Delete documents from Qdrant"""
        try:
            # Delete from Qdrant
            self.client.delete(
                collection_name=self.index_name,
                points_selector=document_ids
            )
            
            # Remove from local metadata store
            for doc_id in document_ids:
                if doc_id in self.metadata_store:
                    del self.metadata_store[doc_id]
            
            logging.info(f"Deleted {len(document_ids)} documents from Qdrant")
            return True
            
        except Exception as e:
            logging.error(f"Error deleting documents from Qdrant: {str(e)}")
            return False
    
    def save(self, path: str) -> bool:
        """Save Qdrant metadata to disk (index is in Qdrant server)"""
        try:
            save_path = Path(path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save metadata
            metadata_path = save_path / f"{self.index_name}_qdrant_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': self.metadata_store,
                    'dimension': self.dimension,
                    'index_name': self.index_name
                }, f)
            
            logging.info(f"Saved Qdrant metadata to {path}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving Qdrant metadata: {str(e)}")
            return False
    
    def load(self, path: str) -> bool:
        """Load Qdrant metadata from disk"""
        try:
            load_path = Path(path)
            
            # Load metadata
            metadata_path = load_path / f"{self.index_name}_qdrant_metadata.json"
            with open(metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.metadata_store = data['metadata']
                self.dimension = data['dimension']
            
            logging.info(f"Loaded Qdrant metadata from {path}")
            return True
            
        except Exception as e:
            logging.error(f"Error loading Qdrant metadata: {str(e)}")
            return False

class VectorDatabaseManager:
    """Manager for handling vector database operations"""
    def __init__(self, db_type: str = "faiss", dimension: int = 384, 
                index_name: str = "conspiracy_theories", config: Dict[str, Any] = None):
        self.db_type = db_type
        self.dimension = dimension
        self.index_name = index_name
        self.config = config or {}
        
        # Initialize appropriate vector database
        if db_type == "faiss":
            if not FAISS_AVAILABLE:
                raise ImportError("FAISS is not available. Install with: pip install faiss-cpu or faiss-gpu")
            self.db = FaissVectorDB(
                dimension=dimension,
                index_name=index_name,
                metric=self.config.get("metric", "cosine")
            )
        elif db_type == "pinecone":
            if not PINECONE_AVAILABLE:
                raise ImportError("Pinecone is not available. Install with: pip install pinecone-client")
            self.db = PineconeVectorDB(
                dimension=dimension,
                index_name=index_name,
                api_key=self.config.get("api_key"),
                environment=self.config.get("environment")
            )
        elif db_type == "qdrant":
            if not QDRANT_AVAILABLE:
                raise ImportError("Qdrant is not available. Install with: pip install qdrant-client")
            self.db = QdrantVectorDB(
                dimension=dimension,
                index_name=index_name,
                url=self.config.get("url"),
                api_key=self.config.get("api_key")
            )
        else:
            raise ValueError(f"Unsupported vector database type: {db_type}")
        
        logging.info(f"Initialized Vector Database Manager with {db_type} backend")
    
    def add_documents(self, document_ids: List[str], embeddings: List[List[float]], 
                     metadata: List[Dict[str, Any]]) -> bool:
        """Add documents to the vector database"""
        return self.db.add_documents(document_ids, embeddings, metadata)
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        return self.db.search(query_embedding, top_k)
    
    def delete(self, document_ids: List[str]) -> bool:
        """Delete documents from the vector database"""
        return self.db.delete(document_ids)
    
    def save(self, path: str) -> bool:
        """Save the vector database to disk"""
        return self.db.save(path)
    
    def load(self, path: str) -> bool:
        """Load the vector database from disk"""
        return self.db.load(path)

def main():
    """CLI for vector database operations"""
    parser = argparse.ArgumentParser(description="Vector Database Manager CLI")
    parser.add_argument("--db-type", type=str, default="faiss", choices=["faiss", "pinecone", "qdrant"],
                        help="Vector database backend to use")
    parser.add_argument("--dimension", type=int, default=384,
                        help="Dimension of the vectors")
    parser.add_argument("--index-name", type=str, default="conspiracy_theories",
                        help="Name of the index or collection")
    parser.add_argument("--save-path", type=str, default="data/vectors",
                        help="Path to save vector database files")
    parser.add_argument("--load-path", type=str, default="data/vectors",
                        help="Path to load vector database files from")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Add command
    add_parser = subparsers.add_parser("add", help="Add documents to the vector database")
    add_parser.add_argument("--docs-file", type=str, required=True,
                          help="JSON file with document IDs, embeddings, and metadata")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for similar documents")
    search_parser.add_argument("--query-vector", type=str, required=True,
                             help="JSON file with query vector")
    search_parser.add_argument("--top-k", type=int, default=5,
                             help="Number of results to return")
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete documents from the vector database")
    delete_parser.add_argument("--doc-ids", type=str, required=True,
                             help="JSON file with document IDs to delete")
    
    # Save command
    save_parser = subparsers.add_parser("save", help="Save the vector database to disk")
    
    # Load command
    load_parser = subparsers.add_parser("load", help="Load the vector database from disk")
    
    args = parser.parse_args()
    
    # Configure database
    config = {}
    if args.db_type == "pinecone":
        config["api_key"] = os.getenv("PINECONE_API_KEY")
        config["environment"] = os.getenv("PINECONE_ENVIRONMENT")
    elif args.db_type == "qdrant":
        config["url"] = os.getenv("QDRANT_URL", "http://localhost:6333")
        config["api_key"] = os.getenv("QDRANT_API_KEY")
    
    # Create manager
    manager = VectorDatabaseManager(
        db_type=args.db_type,
        dimension=args.dimension,
        index_name=args.index_name,
        config=config
    )
    
    # Execute command
    if args.command == "add":
        with open(args.docs_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        result = manager.add_documents(
            document_ids=data.get("ids", []),
            embeddings=data.get("embeddings", []),
            metadata=data.get("metadata", [])
        )
        
        if result:
            print("Documents added successfully")
        else:
            print("Failed to add documents")
            
    elif args.command == "search":
        with open(args.query_vector, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = manager.search(
            query_embedding=data.get("vector", []),
            top_k=args.top_k
        )
        
        print(f"Search results: {json.dumps(results, indent=2)}")
        
    elif args.command == "delete":
        with open(args.doc_ids, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        result = manager.delete(document_ids=data.get("ids", []))
        
        if result:
            print("Documents deleted successfully")
        else:
            print("Failed to delete documents")
            
    elif args.command == "save":
        result = manager.save(args.save_path)
        
        if result:
            print(f"Vector database saved to {args.save_path}")
        else:
            print("Failed to save vector database")
            
    elif args.command == "load":
        result = manager.load(args.load_path)
        
        if result:
            print(f"Vector database loaded from {args.load_path}")
        else:
            print("Failed to load vector database")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()