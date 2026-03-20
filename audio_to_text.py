import os
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from dotenv import load_dotenv

load_dotenv()

# LLM Configuration
LLM_MODEL_NAME = os.getenv('LLM_MODEL', 'meta-llama/Llama-2-7b-chat-hf')
HF_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

llm_pipeline = None


def get_llm_pipeline():
    """
    Lazy load the LLM pipeline
    Note: For production, you might want to use quantized models or API endpoints
    """
    global llm_pipeline
    
    if llm_pipeline is None:
        print(f"Loading LLM model: {LLM_MODEL_NAME}")
        print("Note: This may take several minutes and requires significant memory...")
        
        try:
            # Check if CUDA is available
            device = 0 if torch.cuda.is_available() else -1
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(
                LLM_MODEL_NAME,
                token=HF_TOKEN,
                trust_remote_code=True
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                token=HF_TOKEN,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            llm_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15
            )
            
            print("LLM loaded successfully!")
            
        except Exception as e:
            print(f"Error loading LLM: {e}")
            print("Falling back to a simpler model or mock response...")
            # Fallback: you could use a smaller model or return mock responses
            llm_pipeline = "mock"  # Placeholder for mock mode
    
    return llm_pipeline


def generate_answer(query: str, context_chunks: List[Dict]) -> str:
    """
    Generate answer using LLaMA with retrieved context
    """
    
    if not context_chunks:
        return "No relevant information found in the documents."
    
    # Build context from chunks
    context = "\n\n".join([
        f"[{chunk.get('pdf_name', 'Document')} - Page {chunk['page_number']}]: {chunk['text']}"
        for chunk in context_chunks
    ])
    
    # Create prompt
    prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {query}

Answer: Based on the context provided, """
    
    try:
        pipeline = get_llm_pipeline()
        
        # Mock mode fallback (if model couldn't load)
        if pipeline == "mock":
            sources = []
            for chunk in context_chunks[:3]:
                pdf_name = chunk.get('pdf_name', 'Unknown Document')
                page = chunk.get('page_number', '?')
                text_preview = chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
                sources.append(f"📄 {pdf_name} (Page {page}):\n{text_preview}")
            
            answer = f"**Found relevant information from {len(context_chunks)} chunk(s):**\n\n"
            answer += "\n\n---\n\n".join(sources)
            return answer
        
        # Generate response with LLaMA
        print(f"Generating answer with LLaMA for query: {query}")
        response = pipeline(
            prompt,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )
        
        # Extract generated text
        generated_text = response[0]['generated_text']
        
        # Remove the prompt from the response
        answer = generated_text[len(prompt):].strip()
        
        return answer
        
    except Exception as e:
        print(f"Error generating answer: {e}")
        # Fallback response
        pages = ', '.join([str(c['page_number']) for c in context_chunks[:5]])
        return f"I found relevant information in the documents (Pages {pages}), but encountered an error generating a detailed response. Please try rephrasing your question."


def save_query_log(pdf_id: int, pdf_name: str, query: str, answer: str):
    """
    Save query and answer to database
    """
    from database import get_connection
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO query_logs (pdf_id, pdf_name, user_query, llm_answer)
            VALUES (%s, %s, %s, %s)
        """, (pdf_id, pdf_name, query, answer))
        
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error saving query log: {e}")
        raise
    finally:
        cursor.close()
        conn.close()
