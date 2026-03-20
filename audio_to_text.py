import os
import time
from sentence_transformers import SentenceTransformer
from database import get_connection
from dotenv import load_dotenv
from typing import Optional, List

load_dotenv()

# Initialize embedding model
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
embedding_model = None


def get_embedding_model():
    """Lazy load the embedding model"""
    global embedding_model
    if embedding_model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return embedding_model


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using Gemma/SentenceTransformer
    """
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings.tolist()


def store_embeddings(pdf_id: int, project_code: str, pages_data: List[dict], chunk_size: int = 500):
    """
    Generate and store embeddings for PDF chunks, tagging each row with project_code
    so we can search across all documents within the same project later.
    """
    from pdf_processor import chunk_text
    start_time = time.time()

    conn = get_connection()
    cursor = conn.cursor()
    try:
        all_chunks = []
        chunk_metadata = []

        # Create chunks from each page
        for page_data in pages_data:
            page_num = page_data['page_number']
            page_text = page_data['text']
            if not page_text or not page_text.strip():
                continue

            chunks = chunk_text(page_text, chunk_size=chunk_size)
            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_metadata.append({
                    'page_number': page_num,
                    'text': chunk
                })

        if not all_chunks:
            print("No chunks to process")
            return

        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        embeddings = generate_embeddings(all_chunks)
        embedding_time = time.time() - start_time

        # Store embeddings in database (single commit for the whole batch)
        storage_start = time.time()
        insert_sql = """
            INSERT INTO codecatalyst.embeddings (pdf_id, project_code, page_number, chunk_text, embedding)
            VALUES (%s, %s, %s, %s, %s)
        """
        for metadata, embedding in zip(chunk_metadata, embeddings):
            cursor.execute(
                insert_sql,
                (pdf_id, project_code, metadata['page_number'], metadata['text'], embedding)
            )
        conn.commit()

        storage_time = time.time() - storage_start
        print(f"Stored {len(embeddings)} embeddings successfully")

        # Update PDF metadata with timing information
        from pdf_processor import update_pdf_timings
        update_pdf_timings(pdf_id, embedding_time=embedding_time, storage_time=storage_time)

    except Exception as e:
        conn.rollback()
        print(f"Error storing embeddings: {e}")
        raise
    finally:
        cursor.close()
        conn.close()




from sentence_transformers import SentenceTransformer

def _to_vector_literal(vec: List[float]) -> str:
    # pgvector accepts text like: [0.123, 0.456, ...]
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


# --- helpers: keep near the top of embeddings.py ---
from typing import Optional, List
from database import get_connection

def _to_vector_literal(vec: List[float]) -> str:
    """Format a Python list as a pgvector literal string: [0.123456,0.234567,...]"""
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

def build_tsquery(query: str) -> str:
    """
    Very small normalizer that expands common synonyms/variants so FTS can match natural questions:
    - ICAO  ->  (icao | (international & civil & aviation & organization))
    - practices/practises/practice  -> OR-grouped
    You can extend this dict for your domain.
    """
    q = (query or "").lower().strip()

    expansions = {
        "icao": "(icao | (international & civil & aviation & organization))",
    }

    tokens = q.replace("?", " ").replace(",", " ").split()
    groups: List[str] = []
    for t in tokens:
        if t in ("practices", "practises", "practice"):
            groups.append("(practices | practises | practice)")
        elif t in expansions:
            groups.append(expansions[t])
        else:
            groups.append(t)

    # Join groups with AND; inside a group we already used OR
    return " & ".join(groups) if groups else ""

def search_similar_chunks(
    query: str,
    project_code: Optional[str] = None,
    pdf_id: Optional[int] = None,
    similarity_threshold: float = 0.20,   # now applied AFTER fetch (optional)
    keyword_weight: float = 0.5,
    vector_weight: float = 0.5,
    top_k: int = 10
) -> List[dict]:
    """
    Hybrid search (vector + keyword) that is resilient to natural questions.
    - No SQL WHERE cutoff; we ORDER BY combined_score and LIMIT, then optionally filter in Python.
    - If hybrid returns no rows, do a vector-only fallback within the same scope.
    - Returns pdf_id to support mapping to documents.llm_doc_id.
    """
    print(f"Starting search_similar_chunks with query: {query}, "
          f"project_code: {project_code}, pdf_id: {pdf_id}, "
          f"similarity_threshold: {similarity_threshold}")
    print(f"Weights - keyword_weight: {keyword_weight}, vector_weight: {vector_weight}")

    model = get_embedding_model()  # already defined earlier in this file
    query_embedding = model.encode([query])[0].tolist()
    query_vec = _to_vector_literal(query_embedding)  # reliable ::vector cast
    ts_query = build_tsquery(query)

    conn = get_connection()
    cursor = conn.cursor()
    try:
        # ---------- Project-scoped search across ALL PDFs in the project ----------
        if project_code:
            print("Performing hybrid search within a project (all PDFs under project_code)...")
            cursor.execute("""
                WITH vector_scores AS (
                    SELECT
                        e.embedding_id,
                        e.chunk_text,
                        e.page_number,
                        p.pdf_name,
                        p.pdf_id,
                        1 - (e.embedding <=> %s::vector) AS vector_similarity
                    FROM codecatalyst.embeddings e
                    JOIN codecatalyst.pdfs p ON e.pdf_id = p.pdf_id
                    WHERE e.project_code = %s
                ),
                keyword_scores AS (
                    SELECT
                        e.embedding_id,
                        e.chunk_text,
                        e.page_number,
                        p.pdf_name,
                        p.pdf_id,
                        ts_rank(
                            to_tsvector('english', e.chunk_text),
                            to_tsquery('english', %s)
                        ) AS keyword_score
                    FROM codecatalyst.embeddings e
                    JOIN codecatalyst.pdfs p ON e.pdf_id = p.pdf_id
                    WHERE e.project_code = %s
                )
                SELECT
                    v.chunk_text,
                    v.page_number,
                    v.pdf_name,
                    v.pdf_id,
                    v.vector_similarity,
                    COALESCE(k.keyword_score, 0) AS keyword_score,
                    (%s * v.vector_similarity + %s * COALESCE(k.keyword_score, 0)) AS combined_score
                FROM vector_scores v
                LEFT JOIN keyword_scores k ON v.embedding_id = k.embedding_id
                ORDER BY combined_score DESC
                LIMIT %s
            """, (
                query_vec, project_code,
                ts_query, project_code,
                vector_weight, keyword_weight,
                top_k
            ))

            rows = cursor.fetchall()
            similar_chunks: List[dict] = [
                {
                    'text': r[0],
                    'page_number': r[1],
                    'pdf_name': r[2],
                    'pdf_id': r[3],
                    'vector_similarity': float(r[4]) if r[4] is not None else 0.0,
                    'keyword_score': float(r[5]) if r[5] is not None else 0.0,
                    'combined_score': float(r[6]) if r[6] is not None else 0.0,
                }
                for r in rows
            ]

        # ---------- PDF-scoped search (single PDF) ----------
        elif pdf_id:
            print("Performing hybrid search in specific PDF...")
            cursor.execute("""
                WITH vector_scores AS (
                    SELECT
                        e.embedding_id,
                        e.chunk_text,
                        e.page_number,
                        p.pdf_name,
                        p.pdf_id,
                        1 - (e.embedding <=> %s::vector) AS vector_similarity
                    FROM codecatalyst.embeddings e
                    JOIN codecatalyst.pdfs p ON e.pdf_id = p.pdf_id
                    WHERE e.pdf_id = %s
                ),
                keyword_scores AS (
                    SELECT
                        e.embedding_id,
                        e.chunk_text,
                        e.page_number,
                        p.pdf_name,
                        p.pdf_id,
                        ts_rank(
                            to_tsvector('english', e.chunk_text),
                            to_tsquery('english', %s)
                        ) AS keyword_score
                    FROM codecatalyst.embeddings e
                    JOIN codecatalyst.pdfs p ON e.pdf_id = p.pdf_id
                    WHERE e.pdf_id = %s
                )
                SELECT
                    v.chunk_text,
                    v.page_number,
                    v.pdf_name,
                    v.pdf_id,
                    v.vector_similarity,
                    COALESCE(k.keyword_score, 0) AS keyword_score,
                    (%s * v.vector_similarity + %s * COALESCE(k.keyword_score, 0)) AS combined_score
                FROM vector_scores v
                LEFT JOIN keyword_scores k ON v.embedding_id = k.embedding_id
                ORDER BY combined_score DESC
                LIMIT %s
            """, (
                query_vec, pdf_id,
                ts_query, pdf_id,
                vector_weight, keyword_weight,
                top_k
            ))

            rows = cursor.fetchall()
            similar_chunks = [
                {
                    'text': r[0],
                    'page_number': r[1],
                    'pdf_name': r[2],
                    'pdf_id': r[3],
                    'vector_similarity': float(r[4]) if r[4] is not None else 0.0,
                    'keyword_score': float(r[5]) if r[5] is not None else 0.0,
                    'combined_score': float(r[6]) if r[6] is not None else 0.0,
                }
                for r in rows
            ]

        # ---------- Global search across all PDFs ----------
        else:
            print("Performing hybrid search across all PDFs...")
            cursor.execute("""
                WITH vector_scores AS (
                    SELECT
                        e.embedding_id,
                        e.chunk_text,
                        e.page_number,
                        p.pdf_name,
                        p.pdf_id,
                        1 - (e.embedding <=> %s::vector) AS vector_similarity
                    FROM codecatalyst.embeddings e
                    JOIN codecatalyst.pdfs p ON e.pdf_id = p.pdf_id
                ),
                keyword_scores AS (
                    SELECT
                        e.embedding_id,
                        e.chunk_text,
                        e.page_number,
                        p.pdf_name,
                        p.pdf_id,
                        ts_rank(
                            to_tsvector('english', e.chunk_text),
                            to_tsquery('english', %s)
                        ) AS keyword_score
                    FROM codecatalyst.embeddings e
                    JOIN codecatalyst.pdfs p ON e.pdf_id = p.pdf_id
                )
                SELECT
                    v.chunk_text,
                    v.page_number,
                    v.pdf_name,
                    v.pdf_id,
                    v.vector_similarity,
                    COALESCE(k.keyword_score, 0) AS keyword_score,
                    (%s * v.vector_similarity + %s * COALESCE(k.keyword_score, 0)) AS combined_score
                FROM vector_scores v
                LEFT JOIN keyword_scores k ON v.embedding_id = k.embedding_id
                ORDER BY combined_score DESC
                LIMIT %s
            """, (
                query_vec, query,  # for to_tsquery we pass the normalized query string below if desired
                vector_weight, keyword_weight,
                top_k
            ))

            rows = cursor.fetchall()
            similar_chunks = [
                {
                    'text': r[0],
                    'page_number': r[1],
                    'pdf_name': r[2],
                    'pdf_id': r[3],
                    'vector_similarity': float(r[4]) if r[4] is not None else 0.0,
                    'keyword_score': float(r[5]) if r[5] is not None else 0.0,
                    'combined_score': float(r[6]) if r[6] is not None else 0.0,
                }
                for r in rows
            ]

        # ---------- Optional: apply a gentle cutoff in Python ----------
        if similarity_threshold is not None and similarity_threshold > 0:
            before = len(similar_chunks)
            similar_chunks = [c for c in similar_chunks if c.get('combined_score', 0.0) >= similarity_threshold]
            print(f"Filtered by threshold from {before} -> {len(similar_chunks)}")

        # ---------- Fallback #1: vector-only (project scope), if nothing survived ----------
        if not similar_chunks and project_code:
            print("No hybrid hits. Falling back to vector-only search within project_code...")
            cursor.execute("""
                SELECT
                    e.chunk_text,
                    e.page_number,
                    p.pdf_name,
                    p.pdf_id,
                    1 - (e.embedding <=> %s::vector) AS vector_similarity
                FROM codecatalyst.embeddings e
                JOIN codecatalyst.pdfs p ON e.pdf_id = p.pdf_id
                WHERE e.project_code = %s
                ORDER BY vector_similarity DESC
                LIMIT %s
            """, (query_vec, project_code, top_k))
            rows = cursor.fetchall()
            for r in rows:
                similar_chunks.append({
                    'text': r[0],
                    'page_number': r[1],
                    'pdf_name': r[2],
                    'pdf_id': r[3],
                    'vector_similarity': float(r[4]) if r[4] is not None else 0.0,
                    'keyword_score': 0.0,
                    'combined_score': float(r[4]) if r[4] is not None else 0.0,
                })

        # ---------- Fallback #2: keyword-only (project scope), if still empty ----------
        if not similar_chunks and project_code:
            print("Still empty. Falling back to keyword-only search within project_code...")
            cursor.execute("""
                SELECT
                    e.chunk_text,
                    e.page_number,
                    p.pdf_name,
                    p.pdf_id,
                    0.0 AS vector_similarity,
                    ts_rank(
                        to_tsvector('english', e.chunk_text),
                        to_tsquery('english', %s)
                    ) AS keyword_score
                FROM codecatalyst.embeddings e
                JOIN codecatalyst.pdfs p ON e.pdf_id = p.pdf_id
                WHERE e.project_code = %s
                  AND to_tsvector('english', e.chunk_text) @@ to_tsquery('english', %s)
                ORDER BY keyword_score DESC
                LIMIT %s
            """, (ts_query, project_code, ts_query, top_k))
            rows = cursor.fetchall()
            for r in rows:
                similar_chunks.append({
                    'text': r[0],
                    'page_number': r[1],
                    'pdf_name': r[2],
                    'pdf_id': r[3],
                    'vector_similarity': float(r[4]) if r[4] is not None else 0.0,
                    'keyword_score': float(r[5]) if r[5] is not None else 0.0,
                    'combined_score': float(r[5]) if r[5] is not None else 0.0,
                })

        print(f"Formatted similar chunks: {similar_chunks}")
        return similar_chunks

    finally:
        cursor.close()
        conn.close()


