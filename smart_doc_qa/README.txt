# Smart Document Q&A Assistant

## 1. Project Overview
The Smart Document Q&A Assistant is a powerful Retrieval Augmented Generation (RAG) backend designed to empower users to converse intelligently with large PDF documents. By utilizing Google's advanced Gemini generative models and high-performance vector similarity search, the system can digest multi-page documents, accurately extract specific context, and provide formulated, hallucination-free answers—or comprehensive, structured summaries—based entirely on the uploaded text.

## 2. System Architecture
The backend is highly modular, separating concerns into distinct, scalable components:
- FastAPI Server: The high-performance web framework acting as the entry point, handling HTTP requests, file uploads, request validation, and JSON responses.
- Agent Router: An intelligent intent engine that evaluates user queries. It routes specific factual questions to the RAG pipeline or redirects broad exploratory queries to the Summarization pipeline dynamically.
- RAG Pipeline: The core Retrieval Augmented Generation engine that embeds user queries, searches the FAISS database for the most relevant context, and passes strict bounds to Gemini to construct contextual answers.
- Summarization Pipeline: A scalable Map-Reduce processing engine that handles enormous documents by sharding them into safely sized chunks, generating partial summaries, and consolidating them without exceeding LLM context limits.
- Vector Database (FAISS): A local, persistent Facebook AI Similarity Search index that mathematically maps text chunks as matrices, providing lightning-fast top-K Euclidean distance similarity lookups. 
- Gemini API: Google Generative AI powers both the vectorization (`models/gemini-embedding-001`) and the analytical text generation (`gemini-2.5-flash`).

## 3. Backend Workflow
The complete data timeline scales through the following trajectory natively:
1. PDF Upload: A document is securely uploaded via the `/upload` API endpoint.
2. Text Extraction: The `PyPDF` library parses visual pages into continuous string formats.
3. Chunking: The massive string is broken into small, overlapping token configurations ensuring safe bounds limits dynamically.
4. Embeddings: Text chunks are passed to the Gemini API which returns multidimensional geometric vectors characterizing semantic meaning.
5. FAISS Storage: Vectors and their associated string mappings are added mathematically to the FAISS index arrays and locally saved to disk.
6. Query Retrieval: The user asks a question via `/ask`. The Agent Router computes query embeddings, checks FAISS for closest logical matches, and fetches overlapping textual snippets.
7. Gemini Answer Generation: The matched text snippets are bounded by strict instructions and fed natively back to Gemini to cleanly generate a final contextual response.

## 4. Document Processing
Documents are parsed asynchronously via memory stream bytes. To optimize reading limits, massive text block outputs are iteratively sliced using `tiktoken`. We rely on token-based chunking rather than naïve string division. Crucially, consecutive chunks share a stable 15% token overlap, guaranteeing that context natively wrapping around arbitrary partition boundaries is never accidentally severed or omitted.

## 5. Embeddings
Embeddings convert human-readable text blocks into high-dimensional numerical vectors (arrays of floating-point numbers). This mathematical abstraction maps concepts spatially—meaning sentences with similar contextual meanings reside tightly matched across Euclidean distances geometrically. 
The system leverages Google's specific `models/gemini-embedding-001` to batch process our document chunks, enabling the FAISS index to understand relationships between the abstract user queries and the core underlying document segments efficiently.

## 6. Vector Database
The system incorporates FAISS (Facebook AI Similarity Search) utilizing the `IndexFlatL2` geometric algorithm. FAISS stores all calculated chunk embeddings purely in memory providing extremely rapid L2-distance nearest-neighbor searches natively mapping vectors.
To ensure the service remains fully stateless and resilient, the FAISS indices are serialized natively to a binary configuration (`faiss_index.bin`) alongside an array manifest (`metadata.json`). On API restarts, the application seamlessly re-bootstraps the matrix mapping logic instantly.

## 7. RAG Pipeline
The Retrieval Augmented Generation (RAG) pipeline strictly avoids LLM hallucinations by enforcing grounding constraints natively. When a query is mapped:
1. The pipeline transforms the abstract query into identical geometrical vector parameters.
2. It fetches the top 5 closest text arrays parsed from FAISS.
3. It performs a safety threshold check natively. If the `similarity_score` across nearest embeddings evaluates below 0.65, it securely aborts limits avoiding wild guesses.
4. If successful, boundaries are injected into a strict generative template instructing Gemini to "Answer only from the provided context."

## 8. Summarization Pipeline
For holistic analytical requests, the system invokes a custom Map-Reduce Strategy recursively avoiding restrictive maximum context overlaps natively:
- Path 1 (Direct): If the document strictly evaluates under safe token threshold limits (8000 tokens), it's natively parsed directly cleanly avoiding mapping arrays.
- Path 2 (Map-Reduce): If oversized, it shards across safe lengths, recursively iterating single generative "Partial Summaries" (Map Step), before consolidating all fragmented texts generating a single, globally accurate Unified Summary (Reduce Step). Both limits generate uniform JSON-style headers detailing: Main Topic, Key Points, Important Conclusions.

## 9. Agent Routing Logic
The `agent_router.py` acts as a lightweight heuristic intent-classifier natively tracking incoming queries mapping structures dynamically cleanly. Utilizing Regex mapping bounding limits it analyzes the `query` against explicit triggers (`"summarize", "summary", "overview", "explain document", "brief", "briefly", "key points"`). 
If flagged natively, it overrides general context logic dispatching the entire payload directly cleanly towards the Summarization Pipeline. Otherwise natively mapping normal queries cleanly securely towards RAG vectorization.

## 10. API Endpoints

### POST /upload
Accepts a physical PDF document, chunks limits seamlessly embedding logic natively storing matrix indexing limits natively.
Request: multipart/form-data containing `file`
Response:
{
  "message": "File processed and natively indexed dynamically successfully.",
  "chunks_created": 42
}

### POST /ask
Evaluates string mapping logical inputs tracking dynamic semantic limits routing responses smoothly across logic boundaries dynamically natively.
Request:
{
  "query": "What are the core capabilities of the system?"
}
Response:
{
  "query": "What are the core capabilities of the system?",
  "answer": "The system securely evaluates and parses massive textual limits efficiently routing queries parsing map-reduce bounds safely dynamically natively."
}

## 11. Project Structure

smart_doc_qa/
|
|-- main.py             # FastAPI entry point handling server configurations & routes.
|-- pdf_processor.py    # Responsible for PDF extraction and structural token chunking limits.
|-- embeddings.py       # Interfaces with Gemini API executing chunked vector mappings. 
|-- vector_store.py     # Manages initialising, saving, and querying the local FAISS index.
|-- rag_pipeline.py     # Coordinates strict limits searching FAISS and restricting LLM text generation bounds.
|-- summarizer.py       # Implements the recursive scalable Map-Reduce architectural summarization engine.
|-- agent_router.py     # Heuristically evaluates queries, pushing data towards RAG or Summarizer agents dynamically.
|-- app.py              # Streamlit Frontend mapping APIs graphically cleanly safely locally.
|-- .env                # Safely manages critical limits (e.g. GEMINI_API_KEY).
|-- requirements.txt    # Maps underlying pip dependencies.
|-- README.txt          # Documentation for the complete backend structure.


## 12. Tech Stack
Built utilizing standard best practices tracking optimized explicit dependencies dynamically natively:
- Python / FastAPI: Synchronous robust backend REST routing bounds.
- Gemini API (google-generativeai): Superior LLM embeddings and structural generative analysis natively.
- FAISS (faiss-cpu): High-performance local Euclidean vector search matrices natively.
- PyPDF: Unstructured document PDF stream extraction maps cleanly.
- Tiktoken: Core BPE token limits chunking overlapping texts dynamically mapped.
- Dotenv: Environment boundary protections seamlessly isolated securely.
