import os
import logging
import google.generativeai as genai
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables safely
load_dotenv()

# Initialize Gemini API key dynamically
# Supporting both standard GEMINI_API_KEY and placeholder LLM_API_KEY from .env
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")

if not api_key:
    logger.warning("GEMINI_API_KEY or LLM_API_KEY environment variable is not set. API calls will fail.")
else:
    genai.configure(api_key=api_key)

# The standard recommended embedding model for general text tasks
EMBEDDING_MODEL = "models/gemini-embedding-001"

def generate_embedding(text: str) -> list[float]:
    """
    Generates a vector embedding for a single string of text using Google's Gemini model.
    Handles empty text safely.
    
    Args:
        text (str): The string content to embed.
        
    Returns:
        list[float]: The resulting embedding multidimensional vector.
        
    Raises:
        ValueError: If text is empty.
        Exception: If the API call to Gemini fails.
    """
    try:
        if not text.strip():
            raise ValueError("Cannot generate embedding for empty text.")
            
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document",
            title="Smart Document Q&A Embedding"
        )
        return result['embedding']
    except Exception as e:
        logger.error(f"Failed to generate embedding for single text snippet. Reason: {e}")
        raise

def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generates vector embeddings for a batch of text chunks via the Gemini API.
    
    Args:
        texts (list[str]): A list of string chunks to calculate embeddings for.
        
    Returns:
        list[list[float]]: A list of dimension matrices representing each chunk.
        
    Raises:
        Exception: If the batch API call encounters network or validation limits.
    """
    try:
        # Filter cleanly to prevent sending empty blocks
        valid_texts = [text for text in texts if text.strip()]
        
        if not valid_texts:
            logger.warning("No valid text chunks provided for batch embedding processing.")
            return []

        # Gemini's embed_content natively supports passing a list of strings
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=valid_texts,
            task_type="retrieval_document",
            title="Smart Document Q&A Embedding Batch"
        )
        
        return result["embedding"]
    except Exception as e:
        logger.error(f"Failed to generate batch embeddings array. Reason: {e}")
        raise

def get_embedding_dimension() -> int:
    """
    Helper function to determine the embedding vector dimension.
    Useful for initializing vector stores like FAISS dynamically.
    
    Returns:
        int: The dimension size of the embedding vectors.
    """
    test_vector = generate_embedding("test")
    return len(test_vector)
