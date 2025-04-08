# src/rag/pipeline.py
import logging
import json
import os
import time
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Import local modules
from ..database.vector_database import VectorDatabaseManager
from ..llm.model_manager import LLMManager
from ..factcheck.validator import FactChecker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/rag_pipeline.log"),
        logging.StreamHandler()
    ]
)

class RAGPipeline:
    """
    Retrieval-Augmented Generation Pipeline for conspiracy theory generation
    with fact-checking capabilities.
    """
    def __init__(
        self, 
        vector_db_config: Dict[str, Any],
        llm_config: Dict[str, Any],
        factcheck_config: Dict[str, Any] = None,
        retrieval_config: Dict[str, Any] = None
    ):
        """
        Initialize the RAG Pipeline with vector database, LLM, and fact-checking components.
        
        Args:
            vector_db_config: Configuration for vector database
            llm_config: Configuration for language model
            factcheck_config: Configuration for fact-checking module
            retrieval_config: Configuration for retrieval parameters
        """
        self.retrieval_config = retrieval_config or {
            "top_k": 5,
            "similarity_threshold": 0.7,
            "max_context_length": 2048,
            "use_hybrid_search": False
        }
        
        # Initialize Vector Database
        self.vector_db = VectorDatabaseManager(
            db_type=vector_db_config.get("db_type", "faiss"),
            dimension=vector_db_config.get("dimension", 384),
            index_name=vector_db_config.get("index_name", "conspiracy_theories"),
            config=vector_db_config.get("config", {})
        )
        
        # Initialize LLM
        self.llm = LLMManager(
            model_name=llm_config.get("model_name", "gpt-3.5-turbo"),
            api_key=llm_config.get("api_key"),
            config=llm_config.get("config", {})
        )
        
        # Initialize Fact Checker (optional)
        self.use_factcheck = factcheck_config is not None
        if self.use_factcheck:
            self.factchecker = FactChecker(
                api_key=factcheck_config.get("api_key"),
                model_name=factcheck_config.get("model_name", "gpt-4"),
                config=factcheck_config.get("config", {})
            )
        
        logging.info("RAG Pipeline initialized successfully")
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for user query.
        
        Args:
            query: User query text
            
        Returns:
            Vector embedding of the query
        """
        try:
            # Use the LLM's embedding model to generate query embedding
            embedding = self.llm.generate_embedding(query)
            return embedding
        except Exception as e:
            logging.error(f"Error embedding query: {str(e)}")
            # Return empty embedding in case of error
            return []
    
    def retrieve_documents(self, query_embedding: List[float], filter_criteria: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from the vector database.
        
        Args:
            query_embedding: Embedded query vector
            filter_criteria: Optional filter to apply to search results
            
        Returns:
            List of retrieved documents with metadata
        """
        try:
            # Get top_k documents from vector DB
            top_k = self.retrieval_config["top_k"]
            results = self.vector_db.search(query_embedding, top_k=top_k)
            
            # Apply similarity threshold filtering
            threshold = self.retrieval_config["similarity_threshold"]
            filtered_results = [r for r in results if r["score"] >= threshold]
            
            # Apply additional filters if provided
            if filter_criteria:
                for key, value in filter_criteria.items():
                    filtered_results = [
                        r for r in filtered_results 
                        if key in r["metadata"] and r["metadata"][key] == value
                    ]
            
            logging.info(f"Retrieved {len(filtered_results)} documents from vector database")
            return filtered_results
        except Exception as e:
            logging.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def enrich_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enriches retrieved documents with additional information
        such as validity scores, source credibility, etc.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Enriched documents with additional metadata
        """
        try:
            enriched_docs = []
            
            for doc in documents:
                # Add source credibility score (simplified example)
                source = doc["metadata"].get("source", "unknown")
                
                # Very basic credibility scoring - this should be replaced with a proper system
                credibility_scores = {
                    "academic": 0.9,
                    "news": 0.7,
                    "blog": 0.5,
                    "social_media": 0.3,
                    "unknown": 0.1
                }
                
                credibility = credibility_scores.get(
                    doc["metadata"].get("source_type", "unknown"), 
                    0.1
                )
                
                # Add recency score (1.0 = very recent, 0.0 = very old)
                pub_date = doc["metadata"].get("publication_date", "2000-01-01")
                current_year = 2025  # Update as needed
                try:
                    year = int(pub_date.split("-")[0])
                    recency = min(1.0, max(0.0, (year - 2000) / (current_year - 2000)))
                except:
                    recency = 0.0
                
                # Add enrichment to document
                doc["metadata"]["credibility_score"] = credibility
                doc["metadata"]["recency_score"] = recency
                
                # Calculate overall relevance score (combine retrieval score with other factors)
                relevance = doc["score"] * 0.6 + credibility * 0.2 + recency * 0.2
                doc["relevance_score"] = relevance
                
                enriched_docs.append(doc)
            
            # Sort by relevance
            enriched_docs.sort(key=lambda d: d["relevance_score"], reverse=True)
            
            return enriched_docs
        except Exception as e:
            logging.error(f"Error enriching documents: {str(e)}")
            return documents
    
    def format_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context for the LLM.
        
        Args:
            documents: List of retrieved and enriched documents
            
        Returns:
            Formatted context string
        """
        try:
            context_parts = []
            total_length = 0
            max_length = self.retrieval_config["max_context_length"]
            
            for doc in documents:
                # Format single document
                doc_text = doc["metadata"].get("text", "")
                source = doc["metadata"].get("source", "Unknown Source")
                doc_context = f"Source: {source}\n{doc_text}"
                
                # Check if we're exceeding the max context length
                if total_length + len(doc_context) > max_length:
                    # Truncate if necessary
                    remaining = max_length - total_length
                    if remaining > 100:  # Only add if we can include a meaningful chunk
                        doc_context = doc_context[:remaining]
                        context_parts.append(doc_context)
                    break
                
                context_parts.append(doc_context)
                total_length += len(doc_context)
            
            # Combine all context parts
            combined_context = "\n\n---\n\n".join(context_parts)
            
            logging.info(f"Formatted context with {len(documents)} documents ({total_length} chars)")
            return combined_context
        except Exception as e:
            logging.error(f"Error formatting context: {str(e)}")
            return ""
    
    def generate_response(self, query: str, context: str, creativity_level: float = 0.7) -> str:
        """
        Generate conspiracy theory response using the LLM.
        
        Args:
            query: User query
            context: Retrieved and formatted context
            creativity_level: How creative/speculative the response should be (0.0-1.0)
            
        Returns:
            Generated conspiracy theory
        """
        try:
            # Create prompt for the LLM
            system_prompt = f"""You are a conspiracy theory generator that creates entertaining and engaging conspiracy theories 
based on real information. Your creativity level is set to {creativity_level} (where 0 means strictly factual and 
1 means wildly speculative). Use the provided context to ground your theory, but feel free to make creative connections
between different pieces of information. Your conspiracy theory should be entertaining and thought-provoking.

IMPORTANT: Always clearly label your response as fictional entertainment content.
"""
            
            prompt = f"""
Context information:
{context}

User Query:
{query}

Generate a conspiracy theory that incorporates elements from the context, is entertaining,
and matches the creativity level. Include "FICTIONAL CONSPIRACY THEORY:" at the beginning of your response.
"""
            
            # Generate response
            response = self.llm.generate_text(
                system_prompt=system_prompt,
                prompt=prompt,
                temperature=min(0.5 + creativity_level * 0.5, 0.95),  # Scale temperature based on creativity
                max_tokens=1000
            )
            
            # Ensure the response is labeled as fictional if not already
            if not response.startswith("FICTIONAL CONSPIRACY THEORY:"):
                response = "FICTIONAL CONSPIRACY THEORY:\n" + response
            
            return response
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return "Error generating conspiracy theory."
    
    def fact_check_response(self, response: str, context: str) -> Tuple[str, float]:
        """
        Fact-check the generated response against reliable sources.
        
        Args:
            response: Generated conspiracy theory
            context: Retrieved context used to generate the response
            
        Returns:
            Tuple of (fact_checked_response, validity_score)
        """
        if not self.use_factcheck:
            # If fact-checking is disabled, return as is with a default score
            return response, 0.5
        
        try:
            # Use fact checker to evaluate and annotate the response
            checked_response, validity_score = self.factchecker.check_theory(
                theory=response,
                context=context
            )
            
            logging.info(f"Fact-checked response with validity score: {validity_score}")
            return checked_response, validity_score
        except Exception as e:
            logging.error(f"Error during fact-checking: {str(e)}")
            return response, 0.5
    
    def process_query(
        self, 
        query: str,
        creativity_level: float = 0.7,
        filter_criteria: Dict[str, Any] = None,
        use_fact_check: bool = True
    ) -> Dict[str, Any]:
        """
        Process a user query through the entire RAG pipeline.
        
        Args:
            query: User query text
            creativity_level: Level of creativity for response generation (0.0-1.0)
            filter_criteria: Optional filters for document retrieval
            use_fact_check: Whether to apply fact-checking
            
        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()
        
        # Step 1: Embed the query
        logging.info(f"Processing query: {query}")
        query_embedding = self.embed_query(query)
        
        # Step 2: Retrieve relevant documents
        documents = self.retrieve_documents(query_embedding, filter_criteria)
        
        # Step 3: Enrich documents with additional metadata
        enriched_docs = self.enrich_documents(documents)
        
        # Step 4: Format context for the LLM
        context = self.format_context(enriched_docs)
        
        # Step 5: Generate conspiracy theory response
        response = self.generate_response(query, context, creativity_level)
        
        # Step 6: Fact-check response (if enabled)
        validity_score = 0.5
        if use_fact_check and self.use_factcheck:
            response, validity_score = self.fact_check_response(response, context)
        
        # Prepare the result
        processing_time = time.time() - start_time
        result = {
            "query": query,
            "response": response,
            "validity_score": validity_score,
            "creativity_level": creativity_level,
            "retrieved_documents": len(documents),
            "processing_time_seconds": processing_time,
            "sources": [
                {
                    "title": doc["metadata"].get("title", "Unknown"),
                    "source": doc["metadata"].get("source", "Unknown"),
                    "relevance": doc["relevance_score"]
                }
                for doc in enriched_docs[:3]  # Include top 3 sources
            ]
        }
        
        logging.info(f"Completed query processing in {processing_time:.2f} seconds")
        return result

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading config from {config_path}: {str(e)}")
        return {}

def create_pipeline_from_config(config_path: str) -> RAGPipeline:
    """Create a RAG pipeline from a configuration file"""
    config = load_config(config_path)
    
    pipeline = RAGPipeline(
        vector_db_config=config.get("vector_db", {}),
        llm_config=config.get("llm", {}),
        factcheck_config=config.get("factcheck", None),
        retrieval_config=config.get("retrieval", {})
    )
    
    return pipeline