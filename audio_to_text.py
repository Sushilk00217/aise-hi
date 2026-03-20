import PyPDF2
import time
from typing import List, Dict
from database import get_connection


def extract_pdf_text(pdf_path: str) -> Dict:
    """
    Extract text from PDF page by page.
    Returns a dictionary with page-level text and metadata.
    """
    start_time = time.time()
    pages_text = []
    full_text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                pages_text.append({
                    'page_number': page_num + 1,
                    'text': page_text
                })
                full_text += f"\n--- Page {page_num + 1} ---\n{page_text}"

        extraction_time = time.time() - start_time
        return {
            'pages': pages_text,
            'full_text': full_text,
            'num_pages': num_pages,
            'extraction_time': extraction_time
        }
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        raise


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into chunks with overlap.
    """
    words = text.split()
    chunks = []
    # step size is chunk_size - overlap
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def save_pdf_metadata(pdf_name: str, num_pages: int, full_text: str,
                      extraction_time: float, project_code: str) -> int:
    """
    Save PDF metadata (including project_code) to database and return pdf_id.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO codecatalyst.pdfs (pdf_name, project_code, number_of_pages, full_pdf_text, extraction_time_seconds)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING pdf_id
        """, (pdf_name, project_code, num_pages, full_text, extraction_time))
        pdf_id = cursor.fetchone()[0]
        conn.commit()
        return pdf_id
    except Exception as e:
        conn.rollback()
        print(f"Error saving PDF metadata: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def update_pdf_timings(pdf_id: int, embedding_time: float = None, storage_time: float = None):
    """
    Update PDF processing timings.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        if embedding_time is not None:
            cursor.execute(
                "UPDATE codecatalyst.pdfs SET embedding_time_seconds = %s WHERE pdf_id = %s",
                (embedding_time, pdf_id)
            )
        if storage_time is not None:
            cursor.execute(
                "UPDATE codecatalyst.pdfs SET storage_time_seconds = %s WHERE pdf_id = %s",
                (storage_time, pdf_id)
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error updating PDF timings: {e}")
        raise
    finally:
        cursor.close()
        conn.close()
