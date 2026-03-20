import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os
from dotenv import load_dotenv
import json
from typing import Any, Dict, List, Optional

load_dotenv()

DB_CONFIG = {
    'host': os.getenv('DB_HOST', '192.168.20.62'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'CodeCatalyst'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres')
}


DB_SCHEMA = os.getenv('DB_SCHEMA', 'codecatalyst')


def get_connection():
    """Get database connection with schema search_path set."""
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor() as c:
        # Ensure unqualified names resolve to codecatalyst.*
        c.execute(f"SET search_path TO {DB_SCHEMA}, public;")
    return conn



def init_database():
    """Initialize database and create tables with pgvector extension"""
    
    # First, connect to default postgres database to create our database if needed
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            database='postgres',
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Create database if it doesn't exist
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_CONFIG['database'],))
        if not cursor.fetchone():
            cursor.execute(f"CREATE DATABASE {DB_CONFIG['database']}")
            print(f"Database {DB_CONFIG['database']} created successfully")
        
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error creating database: {e}")
    
    # Now connect to our database and create tables
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Enable pgvector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        # Create PDFs metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pdfs (
                pdf_id SERIAL PRIMARY KEY,
                pdf_name VARCHAR(500) UNIQUE NOT NULL,
                number_of_pages INTEGER NOT NULL,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                extraction_time_seconds FLOAT,
                embedding_time_seconds FLOAT,
                storage_time_seconds FLOAT,
                full_pdf_text TEXT
            )
        """)
        
        # Create embeddings table with pgvector
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                embedding_id SERIAL PRIMARY KEY,
                pdf_id INTEGER REFERENCES pdfs(pdf_id) ON DELETE CASCADE,
                page_number INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                embedding VECTOR(384),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index on embeddings for faster similarity search
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS embeddings_vector_idx 
            ON embeddings USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """)
        
        # Create query logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_logs (
                query_id SERIAL PRIMARY KEY,
                pdf_id INTEGER REFERENCES pdfs(pdf_id) ON DELETE CASCADE,
                pdf_name VARCHAR(500),
                user_query TEXT NOT NULL,
                llm_answer TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create chat sessions table for history (UUID primary key)
        cursor.execute("""
            CREATE EXTENSION IF NOT EXISTS "pgcrypto";
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                session_name VARCHAR(500) DEFAULT 'New Chat',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create chat messages table (UUID foreign key)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                message_id SERIAL PRIMARY KEY,
                session_id UUID REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                pdf_references TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        print("Database initialized successfully with all tables")
        
    except Exception as e:
        conn.rollback()
        print(f"Error initializing database: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def check_pdf_exists(pdf_name: str) -> bool:
    """Check if PDF already exists in database"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT 1 FROM codecatalyst.pdfs WHERE pdf_name = %s", (pdf_name,))
        exists = cursor.fetchone() is not None
        return exists
    finally:
        cursor.close()
        conn.close()


def delete_all_pdfs():
    """Delete all PDFs and their embeddings"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Delete all embeddings (cascade will handle this, but being explicit)
        cursor.execute("DELETE FROM codecatalyst.embeddings")
        cursor.execute("DELETE FROM codecatalyst.query_logs")
        cursor.execute("DELETE FROM codecatalyst.pdfs")
        conn.commit()
        print("All PDFs and embeddings deleted successfully")
    except Exception as e:
        conn.rollback()
        print(f"Error deleting PDFs: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def get_latest_pdf():
    """Get the most recently uploaded PDF"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT pdf_id, pdf_name, number_of_pages, upload_date 
            FROM codecatalyst.pdfs 
            ORDER BY upload_date DESC 
            LIMIT 1
        """)
        result = cursor.fetchone()
        if result:
            return {
                'pdf_id': result[0],
                'pdf_name': result[1],
                'number_of_pages': result[2],
                'upload_date': result[3]
            }
        return None
    finally:
        cursor.close()
        conn.close()


def create_chat_session(session_name: str = "New Chat"):
    """Create a new chat session"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO codecatalyst.chat_sessions (session_name)
            VALUES (%s)
            RETURNING session_id, session_name, created_at
        """, (session_name,))
        result = cursor.fetchone()
        conn.commit()
        return {
            'session_id': result[0],
            'session_name': result[1],
            'created_at': result[2]
        }
    finally:
        cursor.close()
        conn.close()


def get_all_chat_sessions():
    """Get all chat sessions ordered by most recent"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT s.session_id, s.session_name, s.created_at, s.updated_at,
                   COUNT(m.message_id) as message_count
            FROM codecatalyst.chat_sessions s
            LEFT JOIN codecatalyst.chat_messages m ON s.session_id = m.session_id
            GROUP BY s.session_id, s.session_name, s.created_at, s.updated_at
            ORDER BY s.updated_at DESC
        """)
        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                'session_id': row[0],
                'session_name': row[1],
                'created_at': row[2],
                'updated_at': row[3],
                'message_count': row[4]
            })
        return sessions
    finally:
        cursor.close()
        conn.close()


def get_chat_history(session_id: str):
    """Get all messages for a specific chat session"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT message_id, role, content, pdf_references, created_at
            FROM codecatalyst.chat_messages
            WHERE session_id = %s
            ORDER BY created_at ASC
        """, (session_id,))
        messages = []
        for row in cursor.fetchall():
            mid, role, raw_content, pdf_refs, created_at = row
            if role == "assistant":
                # Parse the JSON envelope we saved
                try:
                    parsed = json.loads(raw_content)
                except Exception:
                    parsed = {"answer": raw_content, "sources": [], "pdf_references": pdf_refs}
                messages.append({
                    "message_id": mid,
                    "role": role,
                    "content": parsed,            # object: {answer, sources, pdf_references}
                    "pdf_references": None,       # already inside content
                    "created_at": created_at
                })
            else:
                messages.append({
                    "message_id": mid,
                    "role": role,
                    "content": raw_content,        # plain text for user messages
                    "pdf_references": pdf_refs,
                    "created_at": created_at
                })
        return messages
    finally:
        cursor.close()
        conn.close()


def save_chat_message(
    session_id: str,
    role: str,
    content: Any,                     # string OR dict (for assistant envelope)
    pdf_references: Optional[str] = None,
    *,
    sources: Optional[List[Dict]] = None   # optional kwarg; old callers still work
):
    """Save a chat message to a session.

    - For role == "assistant":
        Stores a canonical JSON envelope in chat_messages.content:
          {"answer": <string|object>, "sources": <list>, "pdf_references": <string|null>}
        You may pass:
          - content as a dict (already an envelope)  -> will be json-dumped as-is
          - content as a string (answer text or JSON string) + sources list -> envelope will be built
    - For other roles: stores content as-is (string).
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        content_to_store = content

        if role == "assistant":
            def _looks_like_json(s: str) -> bool:
                s = (s or "").strip()
                return s.startswith("{") and s.endswith("}")

            if isinstance(content, dict):
                # Already a rich structure: ensure keys exist
                envelope: Dict[str, Any] = {
                    "answer": content.get("answer", content),
                    "sources": content.get("sources", sources or []),
                    "pdf_references": content.get("pdf_references", pdf_references),
                }
            else:
                # content is a string (answer text or possibly a JSON string)
                answer_obj: Any = content
                if isinstance(content, str) and _looks_like_json(content):
                    # Keep the JSON string as-is under "answer"
                    answer_obj = content
                envelope = {
                    "answer": answer_obj,
                    "sources": sources or [],
                    "pdf_references": pdf_references,
                }

            # Persist canonical JSON string
            content_to_store = json.dumps(envelope, ensure_ascii=False)

        cursor.execute("""
            INSERT INTO codecatalyst.chat_messages (session_id, role, content, pdf_references)
            VALUES (%s, %s, %s, %s)
            RETURNING message_id
        """, (session_id, role, content_to_store, pdf_references))
        message_id = cursor.fetchone()[0]

        # Touch session.updated_at
        cursor.execute("""
            UPDATE codecatalyst.chat_sessions
            SET updated_at = CURRENT_TIMESTAMP
            WHERE session_id = %s
        """, (session_id,))

        conn.commit()
        return message_id
    finally:
        cursor.close()
        conn.close()


def update_session_name(session_id: str, new_name: str):
    """Update chat session name"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            UPDATE codecatalyst.chat_sessions 
            SET session_name = %s, updated_at = CURRENT_TIMESTAMP
            WHERE session_id = %s
        """, (new_name, session_id))
        conn.commit()
    finally:
        cursor.close()
        conn.close()


def delete_chat_session(session_id: str):
    """Delete a chat session and all its messages"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("DELETE FROM codecatalyst.chat_sessions WHERE session_id = %s", (session_id,))
        conn.commit()
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    # Initialize database when run directly
    init_database()
