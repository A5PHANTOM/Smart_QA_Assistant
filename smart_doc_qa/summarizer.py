import logging
import tiktoken
import google.generativeai as genai

logger = logging.getLogger(__name__)

# Constants mapping thresholds dynamically
MODEL_NAME = "gemini-2.5-flash"
MAX_TOKEN_LIMIT = 8000  # Conservative limit triggering map-reduce scaling explicitly and preventing overflow

# Initialize model globally to reduce instantiation overhead and improve map-reduce performance
model = genai.GenerativeModel(MODEL_NAME)

# Strict standard summarization prompt structure to ensure native mapping layouts correctly
SUMMARY_PROMPT = """You are an expert document summarizer.
Please read the following text and provide a structured summary containing exactly these three sections:
1. Main Topic: (1-2 sentences summarizing the core subject)
2. Key Points: (Bullet list of the most important facts or arguments)
3. Important Conclusions: (1-3 sentences stating the final verdict or takeaways)

Text to summarize:
{text}
"""

def chunk_text_by_tokens(text: str, max_tokens: int) -> list[str]:
    """
    Helper function breaking massive text bodies into smaller token-safe chunks specifically mapping boundaries.
    Adds ~15% overlap across chunks to ensure context near boundaries isn't lost.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    
    chunks = []
    
    # Calculate step length permitting a 15% overlap window automatically
    STEP = int(max_tokens * 0.85)
    
    for i in range(0, len(tokens), STEP):
        slice_tokens = tokens[i : i + max_tokens]
        chunks.append(encoding.decode(slice_tokens))
        
        # Break early if our slice reached the end
        if i + max_tokens >= len(tokens):
            break
            
    return chunks

def summarize_text_segment(text_segment: str) -> str:
    """
    Core atomic function calling the Gemini model logically enforcing the structural prompt mapping limit boundaries.
    """
    prompt = SUMMARY_PROMPT.format(text=text_segment)
    
    try:
        response = model.generate_content(prompt)
        if response and response.text:
            return response.text.strip()
        return "Summary could not be generated."
    except Exception as e:
        logger.error(f"Failed to generate structured segment summary natively. Error: {e}")
        return "Summary could not be generated."

def summarize_document(text: str) -> str:
    """
    Generates a concise, structured summary of the provided high-level document text dynamically evaluating
    its overall limits natively mapping processing directly onto Gemini capabilities using Map-Reduce patterns.
    
    Args:
        text (str): The full text of the document to be summarized.
        
    Returns:
        str: The structured summary mapping the document natively.
    """
    if not isinstance(text, str):
        raise ValueError("Document text must be a string.")
        
    if not text.strip():
        return "Cannot mathematically summarize an empty document string."
        
    encoding = tiktoken.get_encoding("cl100k_base")
    total_tokens = len(encoding.encode(text))
    
    # Path 1: Generating standard native limits mapping fully inside direct buffer scaling dynamically
    if total_tokens <= MAX_TOKEN_LIMIT:
        logger.info(f"Document size ({total_tokens} tokens) is within conservative limits. Using Direct Summarization.")
        return summarize_text_segment(text)
        
    # Path 2: Map-Reduce Scaling for massive contextual spans dynamically extending array computations 
    logger.info(f"Document size ({total_tokens} tokens) deeply exceeds safety thresholds locally. Initializing Map-Reduce Summarization.")
    
    # Map Step: Break texts into manageable array mappings bounding locally safely mapped constraints dynamically
    text_chunks = chunk_text_by_tokens(text, MAX_TOKEN_LIMIT)
    logger.info(f"Map step: Split massive document natively into {len(text_chunks)} localized segments natively.")
    
    partial_summaries = []
    
    for idx, chunk in enumerate(text_chunks):
        logger.info(f"Processing partial summary for segment mapping explicitly {idx+1}/{len(text_chunks)}...")
        summary = summarize_text_segment(chunk)
        if summary and summary != "Summary could not be generated.":
            partial_summaries.append(f"--- Partial Summary {idx+1} ---\n{summary}")
            
    if not partial_summaries:
        return "Summarization recursively failed natively scaling limits effectively. No valid outputs were returned dynamically."
        
    # Reduce Step: Combine segments and generate the strictly unified final analytical mapping bounds dynamically natively
    combined_partials = "\n\n".join(partial_summaries)
    
    logger.info("Reduce step: Distilling and combining partial outputs natively rendering the final unified summary map.")
    
    final_prompt = f"""You are an expert document summarizer mapped to consolidate fragments.
We have split a very large overarching document mathematically generating subsequent partial summaries. 
Read all the partial structured summaries below, and formulate ONE single unified final Master Summary representing the entire document conceptually correctly globally.

The master summary MUST structurally match this exact schema format seamlessly:
1. Main Topic
2. Key Points
3. Important Conclusions

Combined Partial Summaries:
{combined_partials}
    """
    
    try:
        response = model.generate_content(final_prompt)
        
        if response and response.text:
            logger.info("Map-reduce summarization safely completed bounding structurally successfully.")
            return response.text.strip()
            
        return "Map-reduce unification dynamically failed resolving combined text logic limits."
    except Exception as e:
        logger.error(f"Failed securely executing Reduce step resolving array limits. Error: {e}")
        return "An internal model error happened during final Map-reduce unified distillation logic."
