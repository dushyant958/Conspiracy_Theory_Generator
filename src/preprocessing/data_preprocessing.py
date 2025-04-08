import logging
import json
import time
import argparse
from pathlib import Path
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/preprocessing.log"),
        logging.StreamHandler()
    ]
)

class TextPreprocessor:
    def __init__(self):
        # Download required NLTK resources
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
        except Exception as e:
            logging.warning(f"Error downloading NLTK resources: {str(e)}")
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Add conspiracy-specific stopwords
        self.stop_words.update(['conspiracy', 'theory', 'claim', 'said', 'according'])
    
    def clean_text(self, text):
        """Basic text cleaning"""
        if not isinstance(text, str) or not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """Tokenize text into words"""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """Remove stopwords from token list"""
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize(self, tokens):
        """Lemmatize tokens to their root form"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_text(self, text):
        """Full preprocessing pipeline"""
        if not text:
            return ""
            
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return " ".join(tokens)

class DocumentVectorizer:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Initialize sentence transformer model
        try:
            self.sentence_transformer = SentenceTransformer(model_name)
            logging.info(f"Loaded SentenceTransformer model: {model_name}")
        except Exception as e:
            logging.error(f"Error loading SentenceTransformer model: {str(e)}")
            raise
        
        # Initialize other vectorizers
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        self.count_vectorizer = CountVectorizer(max_features=1000)
    
    def get_embeddings(self, texts, batch_size=32):
        """Generate embeddings using SentenceTransformer"""
        if not texts:
            return np.array([])
            
        try:
            embeddings = []
            # Process in batches to avoid memory issues
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_embeddings = self.sentence_transformer.encode(batch)
                embeddings.append(batch_embeddings)
                
            return np.vstack(embeddings)
        except Exception as e:
            logging.error(f"Error generating embeddings: {str(e)}")
            return np.array([])
    
    def fit_tfidf(self, texts):
        """Fit TF-IDF vectorizer"""
        if not texts:
            return None
            
        try:
            self.tfidf_vectorizer.fit(texts)
            return self.tfidf_vectorizer
        except Exception as e:
            logging.error(f"Error fitting TF-IDF vectorizer: {str(e)}")
            return None
    
    def transform_tfidf(self, texts):
        """Transform texts using fitted TF-IDF vectorizer"""
        if not texts:
            return None
            
        try:
            return self.tfidf_vectorizer.transform(texts)
        except Exception as e:
            logging.error(f"Error transforming with TF-IDF: {str(e)}")
            return None
    
    def fit_transform_count(self, texts):
        """Fit and transform using Count vectorizer"""
        if not texts:
            return None
            
        try:
            return self.count_vectorizer.fit_transform(texts)
        except Exception as e:
            logging.error(f"Error with Count vectorizer: {str(e)}")
            return None

class DataPreprocessingPipeline:
    def __init__(self, input_dir="data/raw", output_dir="data/processed"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.output_dir / "preprocessing_metadata.json"
        
        # Initialize components
        self.text_preprocessor = TextPreprocessor()
        self.vectorizer = DocumentVectorizer()
        
        # Track processing stats
        self.stats = {
            "total_documents": 0,
            "processed_documents": 0,
            "embedding_dimensions": 0,
            "sources_processed": [],
            "errors": 0
        }
    
    def load_data(self, source_filter=None):
        """Load raw data files for processing"""
        data_files = list(self.input_dir.glob("**/*.json"))
        
        if source_filter:
            data_files = [f for f in data_files if any(source in f.name for source in source_filter)]
        
        if not data_files:
            logging.warning(f"No data files found in {self.input_dir}")
            return []
        
        logging.info(f"Found {len(data_files)} data files for processing")
        
        all_documents = []
        for file_path in data_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    documents = json.load(f)
                    
                source_name = file_path.stem.split('_')[0]
                
                # Add source information if not present
                for doc in documents:
                    if isinstance(doc, dict):
                        if 'source' not in doc:
                            doc['source'] = source_name
                        if 'file_origin' not in doc:
                            doc['file_origin'] = str(file_path)
                
                all_documents.extend(documents)
                logging.info(f"Loaded {len(documents)} documents from {file_path}")
                
                if source_name not in self.stats["sources_processed"]:
                    self.stats["sources_processed"].append(source_name)
                
            except Exception as e:
                logging.error(f"Error loading data from {file_path}: {str(e)}")
                self.stats["errors"] += 1
        
        self.stats["total_documents"] = len(all_documents)
        logging.info(f"Loaded {len(all_documents)} documents in total")
        return all_documents
    
    def preprocess_documents(self, documents):
        """Preprocess all documents"""
        processed_documents = []
        
        for i, doc in enumerate(documents):
            try:
                if not isinstance(doc, dict):
                    logging.warning(f"Skipping non-dictionary document at index {i}")
                    continue
                
                # Extract text content based on source type
                text_content = ""
                if 'content' in doc and doc['content']:
                    text_content = doc['content']
                elif 'text' in doc and doc['text']:
                    text_content = doc['text']
                elif 'description' in doc and doc['description']:
                    text_content = doc['description']
                
                # Add title if available
                if 'title' in doc and doc['title']:
                    text_content = f"{doc['title']}. {text_content}"
                
                # Apply text preprocessing
                processed_text = self.text_preprocessor.preprocess_text(text_content)
                
                # Create processed document
                processed_doc = {
                    'id': doc.get('id', f"doc_{i}"),
                    'title': doc.get('title', ''),
                    'source': doc.get('source', 'unknown'),
                    'url': doc.get('url', ''),
                    'original_text': text_content[:1000],  # Truncate for storage
                    'processed_text': processed_text,
                    'metadata': {
                        'processed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'word_count': len(processed_text.split()),
                        'source_type': doc.get('source_type', 'unknown'),
                        'created_date': doc.get('created_date', doc.get('published_at', '')),
                        'origin_file': doc.get('file_origin', '')
                    }
                }
                
                processed_documents.append(processed_doc)
                
                # Log progress
                if (i + 1) % 100 == 0:
                    logging.info(f"Preprocessed {i + 1}/{len(documents)} documents")
                
            except Exception as e:
                logging.error(f"Error preprocessing document at index {i}: {str(e)}")
                self.stats["errors"] += 1
        
        self.stats["processed_documents"] = len(processed_documents)
        logging.info(f"Preprocessing completed. {len(processed_documents)} documents processed.")
        return processed_documents
    
    def vectorize_documents(self, processed_documents):
        """Generate vector embeddings for documents"""
        if not processed_documents:
            logging.warning("No documents to vectorize")
            return processed_documents
        
        # Extract processed text
        texts = [doc['processed_text'] for doc in processed_documents]
        
        logging.info("Generating sentence transformer embeddings...")
        embeddings = self.vectorizer.get_embeddings(texts)
        
        if embeddings.size > 0:
            self.stats["embedding_dimensions"] = embeddings.shape[1]
            
            # Add embeddings to documents
            for i, doc in enumerate(processed_documents):
                doc['embedding'] = embeddings[i].tolist()
            
            logging.info(f"Generated embeddings with {embeddings.shape[1]} dimensions")
        else:
            logging.error("Failed to generate embeddings")
        
        return processed_documents
    
    def save_processed_data(self, processed_documents):
        """Save processed documents and metadata"""
        if not processed_documents:
            logging.warning("No processed documents to save")
            return
        
        # Save in batches to avoid large files
        batch_size = 1000
        for i in range(0, len(processed_documents), batch_size):
            batch = processed_documents[i:i+batch_size]
            
            # Create a version without embeddings for human inspection
            readable_batch = []
            for doc in batch:
                doc_copy = doc.copy()
                if 'embedding' in doc_copy:
                    del doc_copy['embedding']
                readable_batch.append(doc_copy)
            
            # Save readable version
            timestamp = time.strftime('%Y%m%d%H%M%S')
            readable_filename = f"processed_docs_readable_{timestamp}_batch_{i//batch_size}.json"
            readable_filepath = self.output_dir / readable_filename
            
            with open(readable_filepath, 'w', encoding='utf-8') as f:
                json.dump(readable_batch, f, ensure_ascii=False, indent=2)
            
            # Save version with embeddings
            embeddings_filename = f"processed_docs_with_embeddings_{timestamp}_batch_{i//batch_size}.json"
            embeddings_filepath = self.output_dir / embeddings_filename
            
            with open(embeddings_filepath, 'w', encoding='utf-8') as f:
                json.dump(batch, f, ensure_ascii=False)
            
            logging.info(f"Saved batch {i//batch_size + 1} with {len(batch)} documents")
        
        # Save metadata
        self.stats["completion_time"] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Saved preprocessing metadata to {self.metadata_file}")
    
    def run_pipeline(self, source_filter=None):
        """Run the complete preprocessing pipeline"""
        start_time = time.time()
        
        try:
            # Step 1: Load raw data
            documents = self.load_data(source_filter)
            
            # Step 2: Preprocess documents
            processed_documents = self.preprocess_documents(documents)
            
            # Step 3: Generate vector embeddings
            vectorized_documents = self.vectorize_documents(processed_documents)
            
            # Step 4: Save processed data
            self.save_processed_data(vectorized_documents)
            
            # Update and save final stats
            self.stats["total_duration_seconds"] = time.time() - start_time
            self.stats["status"] = "completed"
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)
            
            logging.info(f"Preprocessing pipeline completed in {time.time() - start_time:.2f} seconds")
            return True
            
        except Exception as e:
            self.stats["status"] = "failed"
            self.stats["error"] = str(e)
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)
            
            logging.error(f"Preprocessing pipeline failed: {str(e)}")
            return False

def parse_arguments():
    parser = argparse.ArgumentParser(description="Conspiracy Theory Data Preprocessing")
    parser.add_argument("--input-dir", type=str, default="data/raw",
                        help="Directory containing raw data")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                        help="Directory to store processed data")
    parser.add_argument("--sources", nargs="+", 
                        choices=["wikipedia", "reddit", "news", "government"],
                        help="Specific sources to process")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Create log directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    pipeline = DataPreprocessingPipeline(input_dir=args.input_dir, output_dir=args.output_dir)
    pipeline.run_pipeline(source_filter=args.sources)