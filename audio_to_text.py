from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import shutil
from typing import Optional

from database import (
    init_database, 
    check_pdf_exists, 
    delete_all_pdfs, 
    get_latest_pdf,
    create_chat_session,
    get_all_chat_sessions,
    get_chat_history,
    save_chat_message,
    update_session_name,
    delete_chat_session
)
from pdf_processor import extract_pdf_text, save_pdf_metadata
from embeddings import store_embeddings, search_similar_chunks
from llm import generate_answer, save_query_log

# Initialize FastAPI app
app = FastAPI(title="RAG System API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3003", "http://192.168.20.64:8080"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class QueryRequest(BaseModel):
    query: str
    pdf_id: Optional[int] = None


class ConfirmUploadRequest(BaseModel):
    pdf_name: str
    proceed: bool


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    # Database already initialized - skipping to speed up startup
    # init_database()
    print("RAG System API started successfully")


@app.get("/")
async def root():
    return {"message": "RAG System API is running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/api/upload/check")
async def check_duplicate(file: UploadFile = File(...)):
    """
    Always return false - allow multiple PDFs
    """
    try:
        return {
            "exists": False,
            "pdf_name": file.filename,
            "message": "New PDF detected"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload/process")

@app.post("/api/upload/process")
async def process_pdf(
    file: UploadFile = File(...),
    project_code: str = Form(...),
    force_process: bool = Form(False)
):


    """
    Process uploaded PDF: extract text, generate embeddings, store in database
    """
    try:
        pdf_name = file.filename
        
        # Save uploaded file temporarily
        temp_path = os.path.join(UPLOAD_DIR, pdf_name)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"Processing PDF: {pdf_name}")
        
        # Step 1: Extract PDF text
        print("Step 1: Extracting PDF text...")
        pdf_data = extract_pdf_text(temp_path)
        
        # Step 2: Save PDF metadata
        print("Step 2: Saving PDF metadata...")
        pdf_id = save_pdf_metadata(
    pdf_name=pdf_name,
    num_pages=pdf_data['num_pages'],
    full_text=pdf_data['full_text'],
    extraction_time=pdf_data['extraction_time'],
    project_code=project_code
)
        
        # Step 3: Generate and store embeddings
        print("Step 3: Generating and storing embeddings...")
        store_embeddings(pdf_id, project_code, pdf_data['pages'])        
        # Clean up temporary file
        os.remove(temp_path)
        
        print(f"PDF processing completed successfully. PDF ID: {pdf_id}")
        
        return {
            "success": True,
            "message": "PDF processed successfully",
            "pdf_id": pdf_id,
            "pdf_name": pdf_name,
            "num_pages": pdf_data['num_pages'],
            "extraction_time": pdf_data['extraction_time']
        }
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload/new")
async def upload_new_pdf(file: UploadFile = File(...)):
    """
    Upload new PDF without deleting existing data
    """
    try:
        # Process new PDF without deleting anything
        return await process_pdf(file, force_process=True)
        
    except Exception as e:
        print(f"Error uploading new PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
async def query_pdf(request: QueryRequest):
    """
    Query the PDF using RAG
    """
    try:
        query = request.query
        pdf_id = request.pdf_id
        
        # If no pdf_id provided, get the latest PDF
        if not pdf_id:
            latest_pdf = get_latest_pdf()
            if not latest_pdf:
                raise HTTPException(status_code=404, detail="No PDF found in database")
            pdf_id = latest_pdf['pdf_id']
            pdf_name = latest_pdf['pdf_name']
        else:
            # Get PDF name from database
            from database import get_connection
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT pdf_name FROM codecatalyst.pdfs WHERE pdf_id = %s", (pdf_id,))
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if not result:
                raise HTTPException(status_code=404, detail="PDF not found")
            pdf_name = result[0]
        
        print(f"Processing query: '{query}' for PDF ID: {pdf_id}")
        
        # Step 1: Retrieve similar chunks
        print("Retrieving similar chunks...")
        similar_chunks = search_similar_chunks(query, pdf_id=pdf_id, top_k=5)
        
        if not similar_chunks:
            return {
                "answer": "I couldn't find relevant information in the PDF to answer your question.",
                "sources": []
            }
        
        # Step 2: Generate answer using LLM
        print("Generating answer with LLM...")
        answer = generate_answer(query, similar_chunks)
        
        # Step 3: Save query log
        print("Saving query log...")
        save_query_log(pdf_id, pdf_name, query, answer)
        
        return {
            "answer": answer,
            "sources": [
                {
                    "page_number": chunk['page_number'],
                    "text_preview": chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                    "similarity": chunk['similarity']
                }
                for chunk in similar_chunks
            ],
            "pdf_name": pdf_name
        }
        
    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/latest-pdf")
async def get_latest_pdf_info():
    """
    Get information about the most recently uploaded PDF
    """
    try:
        latest_pdf = get_latest_pdf()
        if not latest_pdf:
            return {"exists": False, "message": "No PDF found"}
        
        return {
            "exists": True,
            "pdf": latest_pdf
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/query-history/{pdf_id}")
async def get_query_history(pdf_id: int, limit: int = 10):
    """
    Get query history for a specific PDF
    """
    try:
        from database import get_connection
        
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT query_id, user_query, llm_answer, timestamp
            FROM codecatalyst.query_logs
            WHERE pdf_id = %s
            ORDER BY timestamp DESC
            LIMIT %s
        """, (pdf_id, limit))
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        history = [
            {
                "query_id": row[0],
                "query": row[1],
                "answer": row[2],
                "timestamp": row[3].isoformat()
            }
            for row in results
        ]
        
        return {"history": history}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/pdfs/all")
async def delete_all_pdfs_endpoint():
    """
    Delete all PDFs and embeddings from the database
    """
    try:
        delete_all_pdfs()
        return {"success": True, "message": "All PDFs and embeddings deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Chat Session Endpoints ====================

@app.post("/api/chat/sessions")
async def create_new_chat_session(session_name: str = "New Chat"):
    """Create a new chat session"""
    try:
        print(f"Creating new chat session: {session_name}")
        session = create_chat_session(session_name)
        print(f"Session created successfully: {session}")
        return session
    except Exception as e:
        print(f"Error creating chat session: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chat/sessions")
async def get_chat_sessions():
    """Get all chat sessions"""
    try:
        sessions = get_all_chat_sessions()
        return {"sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chat/sessions/{session_id}")
async def get_session_history(session_id: str):
    print(f"Received session_id for GET: {session_id}")  # Debug log
    try:
        messages = get_chat_history(session_id)
        return {"messages": messages}
    except Exception as e:
        print(f"Error in get_session_history: {e}")  # Debug log
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/chat/sessions/{session_id}")
async def rename_chat_session(session_id: str, new_name: str):
    """Rename a chat session"""
    try:
        update_session_name(session_id, new_name)
        return {"success": True, "message": "Session renamed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/chat/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    try:
        delete_chat_session(session_id)
        return {"success": True, "message": "Session deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ChatRequest(BaseModel):
    session_id: str
    query: str
    project_code: str

@app.post("/api/chat")
async def chat_with_session(request: ChatRequest):
    print(f"Received session_id for POST: {request.session_id}")
    try:
        query = request.query
        session_id = request.session_id
        project_code = request.project_code  # NEW

        # Basic guard (optional)
        if not project_code or not project_code.strip():
            raise HTTPException(status_code=400, detail="project_code is required")

        # Save user message
        save_chat_message(session_id, "user", query)

        # Project-scoped retrieval across ALL PDFs under the given project
        print(f"Processing query: '{query}' (project_code={project_code})")
        similar_chunks = search_similar_chunks(query, project_code=project_code)
        print(f"similar_chunks for query '{query}': {similar_chunks}")

        if not similar_chunks:
            answer = "I couldn't find relevant information in the documents to answer your question."
            pdf_refs = None
        else:
            # Generate answer
            answer = generate_answer(query, similar_chunks)
            # Build references like: "Spec.pdf (Page 4), Contract.pdf (Page 11)"
            pdf_refs = ", ".join(
                set([f"{chunk.get('pdf_name', 'Unknown')} (Page {chunk['page_number']})"
                     for chunk in similar_chunks])
            )

        # Save assistant message (with refs if any)
        # save_chat_message(session_id, "assistant", answer, pdf_refs)

        save_chat_message(session_id, "assistant", answer, pdf_refs, sources=similar_chunks)
        return {
            "answer": answer,
            "sources": similar_chunks if similar_chunks else [],
            "pdf_references": pdf_refs
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in chat_with_session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
