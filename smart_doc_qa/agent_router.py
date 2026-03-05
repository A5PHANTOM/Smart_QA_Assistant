import re
import logging
# from smart_doc_qa.summarizer import summarize_document
from summarizer import summarize_document
# from smart_doc_qa.rag_pipeline import execute_rag_query
from rag_pipeline import execute_rag_query
logger = logging.getLogger(__name__)

# Pre-defined intent keywords evaluating summarization functional requests
SUMMARIZATION_KEYWORDS = [
    "summarize",
    "summary",
    "overview",
    "explain document",
    "brief",
    "briefly",
    "key points"
]

def route_query(query: str, document_text: str = "") -> str:
    """
    Determines the intent of the user's query and routes it dynamically to the 
    appropriate agent processing block (RAG retrieval or direct document summarization).
    
    Args:
        query (str): The user's specific input query.
        document_text (str): The optional full raw document text (used strictly if summarization intent is detected).
        
    Returns:
        str: The final generated contextual response natively provided by the evaluated sub-agent.
    """
    if not query.strip():
        logger.warning("Empty query received explicitly during routing bounds. Returning error.")
        return "Query cannot be empty."
        
    # Standardize string formatting natively parsing safe equality
    normalized_query = query.lower().strip()
    
    # Evaluate explicit patterns triggering global document summarizations natively,
    # utilizing word boundaries (\b) to avoid false positives (e.g. "briefly" matching "brief")
    is_summarization = any(
        re.search(rf"\b{re.escape(keyword)}\b", normalized_query) 
        for keyword in SUMMARIZATION_KEYWORDS
    )
    
    if is_summarization:
        logger.info("Summarization route selected natively evaluating threshold intent mappings.")
        
        # Guard clause guaranteeing full document text was passed effectively
        if not document_text.strip():
            logger.warning("Summarization route requested but no valid document text parameter was provided to the router explicitly.")
            return "Document text must be provided to effectively generate a summarization."
            
        return summarize_document(text=document_text)
        
    else:
        logger.info("RAG route selected mapping specific dynamic Q&A behaviors.")
        return execute_rag_query(query=query)
