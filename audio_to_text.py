import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

# ✅ BM25 + Hybrid Ensemble Retriever
from langchain_community.retrievers import BM25Retriever  # BM25 integration uses rank_bm25 [1](https://docs.langchain.com/oss/python/integrations/retrievers/bm25)
from langchain_classic.retrievers import EnsembleRetriever       # Hybrid retriever pattern [2](https://apxml.com/courses/langchain-production-llm/chapter-4-production-data-retrieval/hybrid-search-implementation)


# ------------------ SETTINGS ------------------
st.title("📄 Chat with PDFs (Local RAG) — Better Answers (Hybrid: BM25 + Vector)")

PDF_FOLDER = "sats_pdf"
BGE_QUERY_PREFIX = ""  # keep empty for BGE-M3


# ------------------ PROMPTS ------------------
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """You are given chat history and a follow-up question.
Rewrite the follow-up question into a standalone question that is fully self-contained and specific.
Do NOT answer. Keep key entities (names/places/policy titles) intact.

Chat history:
{chat_history}

Follow-up question:
{question}

Standalone question:"""
)

QA_PROMPT = PromptTemplate.from_template(
    """You must answer STRICTLY according to the user's question.

GENERAL RULES:
- Use ONLY the information present in the context.
- Do NOT add background, interpretation, or extra explanation unless the question explicitly asks for it.
- If the answer is not found in the context, say exactly:
  "I couldn't find that in the provided PDFs."

STEP 1 — Identify the QUESTION TYPE:

A) FACTUAL / IDENTIFICATION QUESTIONS
Examples:
- "Which document contains ...?"
- "What is the name of the circular ...?"
- "Which guideline mentions ...?"
- "Where is ... defined?"

➡ If the question is FACTUAL:
- Answer in 1–2 concise sentences.
- Clearly state the exact document name (and circular number if available).
- Do NOT use headings, bullet points, or summaries.
- Do NOT explain purpose, vision, or guidelines unless explicitly asked.

B) EXPLANATORY / OVERVIEW QUESTIONS
Examples:
- "What is ...?"
- "Explain ..."
- "Describe ..."
- "Give an overview of ..."

➡ If the question is EXPLANATORY:
- Write a structured answer using sections and bullet points.
- Summarize all major themes supported by the context.
- Do NOT focus on only one minor detail.

STEP 2 — OUTPUT FORMAT

If FACTUAL question:
- Plain text answer only.

If EXPLANATORY question:
- Use Markdown and headings/bullets.

Context:
{context}

Question:
{question}

Answer:"""
)


# ------------------ RETRIEVER WRAPPER (optional query prefix) ------------------
class QueryPrefixRetriever(BaseRetriever):
    """Wrap any retriever and prefix the query before retrieval."""
    base_retriever: BaseRetriever
    prefix: str = Field(default="")

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        prefixed = f"{self.prefix}{query}"
        if hasattr(self.base_retriever, "invoke"):
            return self.base_retriever.invoke(prefixed)
        return self.base_retriever.get_relevant_documents(prefixed)


# ------------------ LOAD LLM FROM llm_setup.py ------------------
@st.cache_resource
def load_llm_from_setup():
    """
    Expects llm_setup.py to expose either:
      - llm  (a LangChain-compatible LLM instance), OR
      - get_llm() (returns LangChain-compatible LLM instance)
    """
    try:
        from llm_setup import llm
        return llm
    except Exception:
        try:
            from llm_setup import get_llm
            return get_llm()
        except Exception as e:
            raise ImportError(
                "Could not import LLM from llm_setup.py. "
                "Please ensure llm_setup.py defines either `llm` or `get_llm()`."
            ) from e


llm = load_llm_from_setup()


# ------------------ VECTOR STORE (BGE-M3 Embeddings) ------------------
@st.cache_resource
def create_vectorstore_and_chunks():
    documents = []
    if not os.path.isdir(PDF_FOLDER):
        raise FileNotFoundError(f"PDF folder not found: {PDF_FOLDER}")

    for file in os.listdir(PDF_FOLDER):
        if file.lower().endswith(".pdf"):
            path = os.path.join(PDF_FOLDER, file)
            loader = PyPDFLoader(path)
            docs = loader.load()
            for d in docs:
                d.metadata["file_name"] = file
            documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=180,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, chunks


vectorstore, chunks = create_vectorstore_and_chunks()


# ------------------ RETRIEVAL: BM25 + Dense Hybrid -> then Rerank ------------------
@st.cache_resource
def create_hybrid_retriever():
    # 1) Dense retriever (semantic) from FAISS
    dense_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 18, "fetch_k": 60, "lambda_mult": 0.65},
    )

    # 2) BM25 retriever (keyword) over the SAME chunks [1](https://docs.langchain.com/oss/python/integrations/retrievers/bm25)
    bm25_retriever = BM25Retriever.from_documents(chunks, k=18)

    # 3) Hybrid ensemble retriever (RRF weighted merge) [2](https://apxml.com/courses/langchain-production-llm/chapter-4-production-data-retrieval/hybrid-search-implementation)
    hybrid = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.40, 0.60],  # tune as needed
    )

    # 4) Cross-encoder reranker on top (precision booster)
    rerank_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=rerank_model, top_n=6)

    compression_retriever = ContextualCompressionRetriever(
        base_retriever=hybrid,
        base_compressor=compressor
    )

    return QueryPrefixRetriever(base_retriever=compression_retriever, prefix=BGE_QUERY_PREFIX)


retriever = create_hybrid_retriever()


# ------------------ MEMORY + CHAIN ------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    return_source_documents=True,
    output_key="answer"
)


# ------------------ UI ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a question about the PDFs")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    result = qa_chain.invoke({"question": query})
    answer = result.get("answer", "")
    sources = result.get("source_documents", [])

    with st.chat_message("assistant"):
        st.markdown(answer)

        if sources:
            st.markdown("### Sources")
            for d in sources[:8]:
                file_name = d.metadata.get("file_name", "unknown")
                page = d.metadata.get("page", "NA")
                snippet = d.page_content[:280].replace("\n", " ")
                st.write(f"- {file_name}, page {page} — {snippet}...")

    st.session_state.messages.append({"role": "assistant", "content": answer})
