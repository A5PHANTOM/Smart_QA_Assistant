import io
import tiktoken
from pypdf import PdfReader

def extract_text_from_pdf(filepath: str) -> str:
    """
    Extracts raw text content from a given PDF file.
    
    Args:
        filepath (str): The path to the PDF file.
        
    Returns:
        str: The extracted text across all pages.
    """
    reader = PdfReader(filepath)
    text_fragments = []
    
    # Iterate through all pages and extract text
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_fragments.append(page_text)
            
    # Join into a single string
    return "\n".join(text_fragments)

def extract_text_from_pdf_stream(file_stream: io.BytesIO) -> str:
    """
    Extracts raw text content from an uploaded PDF file stream (in-memory).
    Helpful for FastAPI UploadFile integrations.
    
    Args:
        file_stream (io.BytesIO): The uploaded PDF file bytes.
        
    Returns:
        str: The extracted text layout intact.
    """
    reader = PdfReader(file_stream)
    text_fragments = []
    
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_fragments.append(page_text)
            
    return "\n".join(text_fragments)

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 200, encoding_name: str = "cl100k_base") -> list[dict]:
    """
    Splits the extracted text into manageable chunks using a tokenizer.
    
    Why overlap is needed:
    When splitting documents into smaller chunks, a chunk boundary might 
    cut off a sentence or concept midway. By overlapping the end of one chunk 
    with the beginning of the next, we ensure that the contextual meaning around 
    the boundaries is preserved and context isn't lost for the embedding models.
    
    How chunk boundaries are calculated:
    1. The full text is encoded into a list of generic tokens using the model's standard tokenizer.
    2. We iterate through the list of tokens sequentially, taking a literal slice of length `chunk_size`.
    3. The next chunk begins at an index calculated as `current_step - overlap` to 
       ensure `overlap` number of underlying tokens are shared across two sequential chunks.
    4. The token slices are then decoded back into strings to form clean chunk text.
    
    Args:
        text (str): The full text to be chunked.
        chunk_size (int): The maximum size of each chunk (in tokens).
        overlap (int): The number of tokens to overlap between chunks.
        encoding_name (str): The Tiktoken encoding model string to use.
        
    Returns:
        list[dict]: A list of objects containing the index chunk_id and string chunk_text.
    """
    # Validation step 1
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk_size")

    if not text.strip():
        return []

    # Clean excessive whitespace to stabilize tokenization
    text = " ".join(text.split())

    # Get encoding dynamically based on configurations
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception:
        raise ValueError(f"Invalid encoding name: {encoding_name}")
    
    tokens = encoding.encode(text)
    
    chunks = []
    chunk_id = 0
    
    # Iterate through tokens in sliding windows of step size (chunk_size - overlap)
    step_size = max(1, chunk_size - overlap)
    
    for i in range(0, len(tokens), step_size):
        # Slice the token array to form a single chunk of len max `chunk_size`
        chunk_tokens = tokens[i : i + chunk_size]
        
        # Decode the token slice back into natural text
        chunk_str = encoding.decode(chunk_tokens)
        
        # Avoid creating empty chunks that clutter vectors
        if chunk_str.strip():
            chunks.append({
                "chunk_id": chunk_id,
                "chunk_text": chunk_str.strip()
            })
            chunk_id += 1
            
        # Break early if our iteration slice touches or passes the absolute end of the token list
        if i + chunk_size >= len(tokens):
            break
            
    return chunks
