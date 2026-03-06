import os
import io
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

import pdf_processor
import embeddings
import vector_store
import agent_router

# Configure FastAPI application logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define Pydantic models for request validation
class AskRequest(BaseModel):
    query: str

# Initialize unified FastAPI application
app = FastAPI(
    title="Smart Document Q&A Assistant",
    description="API for dynamically parsing multi-page documents structuring answers embedding knowledge locally natively using FAISS + Gemini.",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """
    On server startup: Load FAISS index from disk if it exists.
    Otherwise create a new index mapping correctly initializing structures.
    """
    logger.info("Initializing Qdrant DB connection natively on startup...")
    vector_store.init_db()

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Accepts an uploaded PDF file safely natively evaluating geometry indexing natively storing arrays into FAISS.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Uploaded file limit must strictly be a PDF document.")
        
    try:
        # Read file asynchronously and abstract to in-memory bytes array
        file_bytes = await file.read()
        pdf_stream = io.BytesIO(file_bytes)
        
        # 1. Extract raw text bounds naturally mapping text
        logger.info(f"Extracting textual boundaries parsing bounds natively for '{file.filename}'...")
        full_text = pdf_processor.extract_text_from_pdf_stream(pdf_stream)
        
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="The PDF appears to be empty or effectively unreadable.")
            
        # 2. Chunk text natively parsing specific logical configurations limits safely natively
        logger.info("Chunking text limits bounding token overlaps structurally...")
        chunks = pdf_processor.chunk_text(full_text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Text was safely extracted natively but mathematically produced zero parseable bounds.")
            
        # 3. Generate unified vector arrays natively passing strings cleanly directly natively
        logger.info(f"Generating embeddings natively for {len(chunks)} contextual chunk geometries...")
        chunk_texts = [chunk["chunk_text"] for chunk in chunks]
        doc_embeddings = embeddings.generate_embeddings(chunk_texts)
        
        # 4. Relying on Qdrant handling mathematically
        logger.info(f"Checking Qdrant DB natively tracking collections indexing parameters bindings...")
        
        # Check explicit configurations establishing mapping natively locally dynamically 
        try:
            vector_store.client.get_collection(collection_name=vector_store.COLLECTION_NAME)
        except Exception:
            dimension = embeddings.get_embedding_dimension()
            logger.info(f"Dynamically instantiating missing pristine collection bounds mapping (dim={dimension})...")
            vector_store.init_db(dimension=dimension)
            
        # 5. Store arrays mapping explicitly bounds constraints safely handling geometric configurations
        logger.info("Updating native underlying bounds array storing chunks metadata mapping natively...")
        vector_store.add_documents(chunks, doc_embeddings)
        
        # 6. Save explicitly logic states implicitly mapping natively successfully
        logger.info("Qdrant DB automatically persists point vectors to disk securely.")
        
        return {
            "message": "File processed and natively indexed dynamically successfully.",
            "chunks_created": len(chunks)
        }
    except Exception as e:
        logger.error(f"Error handling /upload endpoint dynamically. Details: {e}")
        raise HTTPException(status_code=500, detail=f"Internal processing logic natively generated error: {e}")

@app.post("/ask")
async def ask_question(request: AskRequest):
    """
    Accepts JSON formatting string constraints pushing queries evaluating outputs through analytical limits routing logic dynamically. 
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="User queried mapping structure natively cannot equal empty boundaries.")
        
    try:
        query = request.query
        
        # When evaluating summarization models cleanly across local bounds, the router asks globally specifically resolving complete parameters dynamically
        # Since Qdrant isolates all loaded target values natively across vectors arrays tracking, we join logically natively here:
        all_texts = vector_store.get_all_document_texts()
        if all_texts:
            document_text = "\n".join(all_texts)
        else:
            document_text = ""
            
        # Push through explicit unified mapping structural agent evaluator
        logger.info(f"Executing ask pipeline cleanly routing abstract structural behaviors resolving limits securely natively for: '{query}'")
        answer = agent_router.route_query(query, document_text=document_text)
        
        return {
            "query": query,
            "answer": answer
        }
    except Exception as e:
        logger.error(f"Error handling limits across boundaries executing native string abstractions seamlessly. Detail: {e}")
        raise HTTPException(status_code=500, detail=f"Internal agent router formatting logic mapped boundary constraint securely: {e}")
