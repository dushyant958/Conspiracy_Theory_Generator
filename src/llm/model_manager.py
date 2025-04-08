import logging
import os
import time
from typing import List, Dict, Any, Optional
import openai
import tiktoken
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/llm_module.log"),
        logging.StreamHandler()
    ]
)

class LLMManager:
    """
    Manages interactions with Large Language Models for text generation
    and embedding creation.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the LLM Manager with model configuration.
        
        Args:
            model_name: Name of the LLM model to use
            api_key: API key for the LLM provider
            config: Additional configuration parameters
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.config = config or {}
        
        # Default embedding model
        self.embedding_model = self.config.get("embedding_model", "text-embedding-ada-002")
        
        # Set max tokens if not specified
        self.max_tokens = self.config.get("max_tokens", 1000)
        
        # Configure OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model_name)
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Fallback encoding
        
        logging.info(f"LLM Manager initialized with model: {self.model_name}")
    
    def _count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))
    
    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        retries: int = 3
    ) -> str:
        """
        Generate text using the configured LLM.
        
        Args:
            prompt: User prompt for text generation
            system_prompt: System instructions for model behavior
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            stop_sequences: Sequences that stop generation
            retries: Number of retry attempts on failure
            
        Returns:
            Generated text
        """
        if max_tokens is None:
            max_tokens = self.max_tokens
            
        system_content = system_prompt or "You are a helpful assistant that creates entertaining conspiracy theories based on provided information."
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ]
        
        attempt = 0
        while attempt < retries:
            try:
                start_time = time.time()
                
                # Calculate token usage for logging
                system_tokens = self._count_tokens(system_content)
                prompt_tokens = self._count_tokens(prompt)
                
                logging.info(f"Generating text with {system_tokens + prompt_tokens} input tokens")
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop_sequences,
                    top_p=0.95,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                
                text = response.choices[0].message.content.strip()
                completion_tokens = self._count_tokens(text)
                
                elapsed_time = time.time() - start_time
                logging.info(f"Generated {completion_tokens} tokens in {elapsed_time:.2f}s")
                
                return text
                
            except Exception as e:
                attempt += 1
                wait_time = 2 ** attempt  # Exponential backoff
                logging.error(f"Error generating text (attempt {attempt}/{retries}): {str(e)}")
                
                if attempt < retries:
                    logging.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logging.error("Max retries reached, returning error message")
                    return "Error generating text. Please try again later."
    
    def generate_embedding(self, text: str, retries: int = 3) -> List[float]:
        """
        Generate embedding vector for a text input.
        
        Args:
            text: Input text to embed
            retries: Number of retry attempts on failure
            
        Returns:
            Vector embedding as list of floats
        """
        attempt = 0
        while attempt < retries:
            try:
                start_time = time.time()
                
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                
                embedding = response.data[0].embedding
                elapsed_time = time.time() - start_time
                
                logging.info(f"Generated embedding in {elapsed_time:.2f}s")
                return embedding
                
            except Exception as e:
                attempt += 1
                wait_time = 2 ** attempt  # Exponential backoff
                logging.error(f"Error generating embedding (attempt {attempt}/{retries}): {str(e)}")
                
                if attempt < retries:
                    logging.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logging.error("Max retries reached, returning empty embedding")
                    return []
    
    def estimate_cost(self, text_length: int, is_embedding: bool = False) -> float:
        """
        Estimate the cost of an API call.
        
        Args:
            text_length: Length of text in tokens
            is_embedding: Whether this is an embedding request
            
        Returns:
            Estimated cost in USD
        """
        # Example pricing (update as needed)
        if is_embedding:
            # Ada embedding pricing
            return (text_length / 1000) * 0.0001
        elif self.model_name == "gpt-4":
            # GPT-4 pricing
            input_cost = (text_length / 1000) * 0.03
            output_cost = (self.max_tokens / 1000) * 0.06
            return input_cost + output_cost
        elif "gpt-3.5" in self.model_name:
            # GPT-3.5 Turbo pricing
            input_cost = (text_length / 1000) * 0.0015
            output_cost = (self.max_tokens / 1000) * 0.002
            return input_cost + output_cost
        else:
            # Default estimate
            return (text_length / 1000) * 0.005