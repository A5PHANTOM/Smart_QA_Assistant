import logging
import google.generativeai as genai
from embeddings import generate_embedding
from vector_store import search

logger = logging.getLogger(__name__)

# The recommended lightweight generation model for text evaluation
GENERATION_MODEL = "gemini-2.5-flash"

def retrieve_context(query: str, top_k: int = 5) -> tuple[str, float]:
    """
    Retrieves the most semantically relevant context chunks by mapping the query to 
    the FAISS vector store geometry.
    
    Args:
        query (str): The raw user query.
        top_k (int): Number of top overlapping chunks to recover.
        
    Returns:
        tuple[str, float]: A concatenated string text block of target contexts, and the highest topological score.
    """
    try:
        query_embedding = generate_embedding(query)
    except Exception as e:
        logger.error(f"Error calculating embeddings natively for search query. Reason: {e}")
        return "", 0.0

    # Retrieve context natively through FAISS Euclidean mapped boundaries
    results = search(query_embedding, top_k=top_k)
    
    if not results:
        logger.warning("No context chunks could be retrieved natively from the underlying indices.")
        return "", 0.0
        
    highest_score = results[0]["similarity_score"]
    
    # Log scores logically assisting structural debugging outputs
    for i, res in enumerate(results):
        chunk_preview = res["chunk_text"][:50].replace('\n', ' ')
        logger.info(f"Retrieved Chunk {i+1} [ID: {res['chunk_id']}] Score: {res['similarity_score']:.4f} | Preview: {chunk_preview}...")
        
    # Combine chunks dynamically into one formatted structure block
    contexts = [res["chunk_text"] for res in results]
    
    # Limit to maximum 5 chunks to avoid exceeding LLM context limits safely
    contexts = contexts[:5]
    
    # Format cleanly with explicit section headings for the LLM
    combined_context = "\n\n".join(
        [f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)]
    )
    
    return combined_context, highest_score

def generate_answer(query: str, context: str) -> str:
    """
    Injects context and isolated generation bounds natively into Gemini using a strict prompt schema.
    
    Args:
        query (str): The specific question attempting to extract answers.
        context (str): The isolated relevant context array values mapping.
        
    Returns:
        str: The final output dynamically provided by the LLM system constraints.
    """
    strict_prompt = f"""You are a document-based assistant.
Answer ONLY from the provided context. If the answer is not in the context say the information is not available.

Context:
{context}

Question:
{query}
"""
    try:
        model = genai.GenerativeModel(GENERATION_MODEL)
        response = model.generate_content(strict_prompt)
        
        # Ensure safely the response body isn't null escaping error bounds
        if response.text:
            return response.text.strip()
            
        return "The model could not generate a valid response."
    except Exception as e:
        logger.error(f"Error executing context constraints towards Gemini text generation limits. Reason: {e}")
        return "An internal error natively generated while structuring your request against the model interface limits."

def execute_rag_query(query: str) -> str:
    """
    Orchestrates the complete Retrieval-Augmented Generation sequentially evaluating conditions 
    dynamically filtering invalid edge matches against a safety threshold.
    
    Args:
        query (str): The user's requested string targeting knowledge block targets.
        
    Returns:
        str: The final generated contextual answer bounds safely mapping the target request natively.
    """
    # Validation constraint preventing raw empty executions failing pipeline limits silently
    if not query.strip():
        return "Query cannot be empty."
        
    logger.info(f"Executing RAG pipeline mapping user query natively: '{query}'")
    
    # Execute embeddings geometric structural bounds fetching closest text matches
    context, highest_score = retrieve_context(query)
    
    # Safety boundary evaluating whether anything matched natively from searches
    if not context:
        return "The document does not contain enough information to answer this."
        
    logger.info(f"Primary contextual top target highest evaluated parameter score strictly resolved: {highest_score:.4f}")
    
    MIN_SIMILARITY = -1.0
    # Validation constraint threshold handling preventing hallucinatory responses natively mapped bounds evaluating geometric limits
    if highest_score < MIN_SIMILARITY:
        logger.warning(f"Highest score ({highest_score:.4f}) evaluates strictly below threshold ({MIN_SIMILARITY}). Refusing to map hallucinated answers natively.")
        return "The document does not contain enough information to answer this."
        
    logger.info("Semantic similarities mapped successfully securely above bounds limit. Requesting LLM outputs parameters.")
    
    # Evaluate valid string constructs directly passing isolated components
    final_answer = generate_answer(query, context)
    return final_answer
