import os
import json
import logging
import numpy as np
import faiss

logger = logging.getLogger(__name__)

# Constants for storage
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.json"

# Global references tracking the index and associated textual chunks
index = None
metadata = []

def initialize_index(dimension: int):
    """
    Initializes a new FAISS index or loads an existing one natively from disk.
    
    FAISS Indexing and Similarity Search:
    - FAISS (Facebook AI Similarity Search) is a library that allows us to quickly search 
      for multimedia documents that are similar to each other natively through vector space.
    - We use IndexFlatL2, which calculates the exact L2 (Euclidean) distance between vectors.
    - A lower computed distance functionally equates to a higher similarity.
    """
    global index, metadata
    
    # Check if we have an existing index to load seamlessly
    if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
        logger.info(f"Existing FAISS index found. Loading from {INDEX_FILE}...")
        load_index()
    else:
        logger.info(f"Creating new FAISS index with dimension {dimension}...")
        # Create an standard L2 distance-based flat index (exact search)
        index = faiss.IndexFlatL2(dimension)
        metadata = []

def add_documents(chunks: list[dict], embeddings: list[list[float]]):
    """
    Adds embeddings to the FAISS index and stores the associated metadata synchronously.
    
    Args:
        chunks (list[dict]): A list of dictionaries containing 'chunk_id' and 'chunk_text'.
        embeddings (list[list[float]]): The vector embeddings corresponding to the chunks.
    """
    global index, metadata
    
    if index is None:
        raise ValueError("FAISS index is not initialized. Call initialize_index() first.")
        
    if len(chunks) != len(embeddings):
        raise ValueError("The number of chunks must match the number of embeddings precisely.")
        
    if not embeddings:
        logger.warning("No embeddings provided to add to the index.")
        return
        
    # Convert pure Python lists into a numpy array (float32 is strictly required by FAISS implementations)
    embeddings_array = np.array(embeddings).astype('float32')
    
    # Add vectors securely to the FAISS index
    index.add(embeddings_array)
    
    # Append the respective metadata sequentially supporting simple position mappings
    metadata.extend(chunks)
    
    logger.info(f"Added {len(chunks)} documents seamlessly to the FAISS index.")
    
    # Automatically save the index after adding documents to persist states actively
    save_index()

def save_index():
    """
    Persists the FAISS index computations and the associated dynamic metadata directly to disk.
    """
    global index, metadata
    
    if index is None:
        logger.warning("Attempted to save an uninitialized index. Skipping.")
        return
        
    # Write the FAISS binary index correctly to disk bounds
    faiss.write_index(index, INDEX_FILE)
    
    # Save the chunk metadata sequentially mapping back to a structured JSON file
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
        
    logger.info("Successfully saved structured FAISS index and metadata to disk.")

def load_index():
    """
    Loads the active FAISS index matrices and stored metadata from disk functionally into memory.
    """
    global index, metadata
    
    if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        raise FileNotFoundError("FAISS index or metadata file not found locally. Ensure they exist before loading requests.")
        
    # Construct the FAISS binary index parser
    index = faiss.read_index(INDEX_FILE)
    
    # Load the synchronized chunk metadata tracking indices
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        
    logger.info(f"Successfully loaded FAISS index mapped bounds tracking {index.ntotal} vectors.")

def search(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """
    Searches the FAISS index accurately for the closest matches mapped to a generic query embedding.
    
    How it works:
    1. The abstract single query embedding is formatted into a 2D structured numpy array.
    2. FAISS natively computes the localized L2 distance mapping between the query vector and all indexed vectors efficiently.
    3. It returns the generic top K closest vector indices cleanly paired alongside their corresponding geometric distances.
    4. We match computed indices efficiently back onto our saved metadata sequentially tracking chunk origins natively.
    5. Discovered L2 distances are mathematically isolated converting values towards a standard semantic similarity score formatting (1 / (1 + distance)) naturally framing higher outputs as best fits securely.
    
    Args:
        query_embedding (list[float]): The float vector geometric representation of the active search query.
        top_k (int): The parameter of top nearest closest matches bounded to be analyzed and returned.
        
    Returns:
        list[dict]: A functional list of localized objects tracking string chunk_text, mapped chunk_id, and similarity_score bounds dynamically.
    """
    global index, metadata
    
    if index is None:
        raise ValueError("FAISS index is not initialized contextually. Call initialize_index() prior to fetching values.")
        
    if index.ntotal == 0:
        logger.warning("FAISS index matrix is explicitly empty. Cannot securely perform comparative mathematical search algorithms.")
        return []
        
    if not query_embedding:
        raise ValueError("Query embedding cannot be empty.")
        
    # Convert the query to a configured 2D numpy array natively for FAISS processing limits (1 exact query, dimensions tracking map length natively)
    query_array = np.array([query_embedding]).astype('float32')
    
    # Calculate the native search computation via the FAISS index library structures: securely mapping distances aligned alongside paired indexed outputs
    distances, indices = index.search(query_array, top_k)
    
    results = []
    
    # Mathematical search structures universally return natively 2D maps, rendering simple array list iterators across target queries optimally processing elements functionally via native `indices[0]` block
    for i in range(len(indices[0])):
        idx = indices[0][i]
        
        # When indices map to empty vectors mapped as defaults `-1` bounding limits, FAISS couldn't sufficiently resolve index matches handling constraints reaching targets mathematically. Standard early break.
        if idx == -1:
            break
            
        # Optional validation handling corrupted edge boundary indexing preventing unverified bounds parsing arrays globally
        if idx >= len(metadata):
            continue
            
        distance = float(distances[0][i])
        
        # We invert the metric since natively L2 distances represent 0.0 as fully identical matching parameters whereas text indexing standardizes scores representing positive correlations logically mapping limits closer towards 1 mathematically globally
        similarity = 1 / (1 + distance)
        
        chunk_data = metadata[idx]
        
        results.append({
            "chunk_text": chunk_data["chunk_text"],
            "chunk_id": chunk_data["chunk_id"],
            "similarity_score": similarity
        })
        
    return results
