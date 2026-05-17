"""
EduGen_AI – utils.py
Backend logic: PDF extraction, chunking, FAISS vector store,
RAG chain, summarization, MCQ generation, key point extraction.
Powered by Groq API (llama-3.3-70b-versatile)
"""

import os
from typing import List, Tuple

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate


# ── Constants ─────────────────────────────────────────────────────────────────
CHUNK_SIZE         = 1500
CHUNK_OVERLAP      = 200
GROQ_MODEL         = "llama-3.3-70b-versatile"   # fast + free on Groq
EMBEDDING_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"  # local, free
MAX_RETRIEVAL_DOCS = 5


# ══════════════════════════════════════════════════════════════════════════════
# PDF TEXT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdfs(uploaded_files) -> str:
    """
    Extract and concatenate text from one or more uploaded PDF files.

    Args:
        uploaded_files: list of Streamlit UploadedFile objects

    Returns:
        str: Combined extracted text from all PDFs
    """
    all_text = []
    for uploaded_file in uploaded_files:
        try:
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    all_text.append(page_text)
        except Exception as e:
            raise RuntimeError(f"Failed to read '{uploaded_file.name}': {e}") from e
    return "\n\n".join(all_text)


# ══════════════════════════════════════════════════════════════════════════════
# CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

def chunk_text(raw_text: str) -> List[str]:
    """Split raw text into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(raw_text)


# ══════════════════════════════════════════════════════════════════════════════
# VECTOR STORE  (uses free local HuggingFace embeddings — no extra API needed)
# ══════════════════════════════════════════════════════════════════════════════

def create_vector_store(raw_text: str, api_key: str) -> FAISS:
    """
    Chunk text, embed with HuggingFace sentence-transformers (free, local),
    and store in FAISS.

    Note: Embeddings are created locally — only the LLM calls use Groq API.

    Args:
        raw_text: Full text extracted from PDFs
        api_key:  Groq API key (stored but not used for embeddings)

    Returns:
        FAISS vector store
    """
    os.environ["GROQ_API_KEY"] = api_key

    chunks = chunk_text(raw_text)
    if not chunks:
        raise ValueError("No text chunks created. The PDF may be empty or image-only.")

    # Free local embeddings — no API key needed for this step
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store


# ══════════════════════════════════════════════════════════════════════════════
# QA CHAIN
# ══════════════════════════════════════════════════════════════════════════════

def get_qa_chain(vector_store: FAISS, api_key: str) -> ConversationalRetrievalChain:
    """
    Build a ConversationalRetrievalChain using Groq LLM.

    Args:
        vector_store: FAISS vector store with indexed document chunks
        api_key:      Groq API key

    Returns:
        ConversationalRetrievalChain ready for invoke()
    """
    os.environ["GROQ_API_KEY"] = api_key

    llm = ChatGroq(
        model=GROQ_MODEL,
        groq_api_key=api_key,
        temperature=0.3,
        max_tokens=1024,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": MAX_RETRIEVAL_DOCS},
    )

    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are EduGen_AI, an expert academic study assistant.
Give accurate, well-structured, helpful answers based ONLY on the document context below.

If the answer is not in the context, say:
"I couldn't find specific information about this in the uploaded documents."

Use bullet points or numbered lists where appropriate. Be concise but thorough.

Context from documents:
{context}

Question: {question}

Detailed Answer:""",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        verbose=False,
    )
    return chain


# ══════════════════════════════════════════════════════════════════════════════
# CHAT HISTORY FORMATTER
# ══════════════════════════════════════════════════════════════════════════════

def format_chat_history(chat_history: List[dict]) -> List[Tuple[str, str]]:
    """
    Convert list of {"role": ..., "content": ...} dicts to (human, ai) tuples
    expected by ConversationalRetrievalChain.
    """
    pairs: List[Tuple[str, str]] = []
    human_msg = None
    for msg in chat_history:
        if msg["role"] == "user":
            human_msg = msg["content"]
        elif msg["role"] == "assistant" and human_msg is not None:
            pairs.append((human_msg, msg["content"]))
            human_msg = None
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARIZATION
# ══════════════════════════════════════════════════════════════════════════════

def summarize_text(raw_text: str, api_key: str, detail_level: str = "standard") -> str:
    """
    Generate a structured summary of the document text using Groq.

    Args:
        raw_text:     Full extracted document text
        api_key:      Groq API key
        detail_level: "brief" | "standard" | "detailed"

    Returns:
        Formatted summary string
    """
    os.environ["GROQ_API_KEY"] = api_key

    detail_instructions = {
        "brief": "Write a concise summary in 3-5 sentences.",
        "standard": (
            "Write a well-structured summary with: "
            "(1) Overview paragraph, "
            "(2) Main Topics covered, "
            "(3) Key Conclusions."
        ),
        "detailed": (
            "Write a comprehensive summary including: "
            "(1) Executive Overview, "
            "(2) All major topics and subtopics with explanations, "
            "(3) Important facts, figures, and definitions, "
            "(4) Conclusions and takeaways."
        ),
    }

    instruction = detail_instructions.get(detail_level, detail_instructions["standard"])
    text_excerpt = raw_text[:8000] if len(raw_text) > 8000 else raw_text

    prompt = f"""You are an expert academic summarizer.
Summarize the following document content.

{instruction}

Use clear headings, bullet points where helpful, and plain English.
Do NOT add information that is not in the document.

Document Content:
{text_excerpt}

Summary:"""

    llm = ChatGroq(
        model=GROQ_MODEL,
        groq_api_key=api_key,
        temperature=0.2,
        max_tokens=1500,
    )
    response = llm.invoke(prompt)
    return response.content


# ══════════════════════════════════════════════════════════════════════════════
# MCQ GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_mcqs(
    raw_text: str,
    api_key: str,
    num_questions: int = 5,
    difficulty: str = "medium",
) -> str:
    """
    Generate multiple-choice questions from document content using Groq.

    Args:
        raw_text:      Full extracted document text
        api_key:       Groq API key
        num_questions: How many MCQs to generate
        difficulty:    "easy" | "medium" | "hard"

    Returns:
        Formatted MCQ string
    """
    os.environ["GROQ_API_KEY"] = api_key

    difficulty_desc = {
        "easy":   "straightforward recall questions suitable for beginners",
        "medium": "application and comprehension questions that require understanding",
        "hard":   "analysis and evaluation questions that require deep understanding",
    }
    desc = difficulty_desc.get(difficulty, difficulty_desc["medium"])
    text_excerpt = raw_text[:7000] if len(raw_text) > 7000 else raw_text

    prompt = f"""You are an expert educator and question designer.
Generate exactly {num_questions} multiple-choice questions (MCQs).
Difficulty: {difficulty.upper()} – {desc}.

Format EACH question like this:

Q[number]. [Question text]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
✅ Correct Answer: [Letter]) [Full correct option]
💡 Explanation: [Brief explanation]

---

Rules:
- Base all questions strictly on the document content.
- Make distractors plausible but clearly incorrect.
- Ensure only ONE correct answer per question.
- Cover different sections of the document.

Document Content:
{text_excerpt}

MCQs:"""

    llm = ChatGroq(
        model=GROQ_MODEL,
        groq_api_key=api_key,
        temperature=0.4,
        max_tokens=2000,
    )
    response = llm.invoke(prompt)
    return response.content


# ══════════════════════════════════════════════════════════════════════════════
# KEY POINT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_key_points(raw_text: str, api_key: str, num_points: int = 10) -> str:
    """
    Extract the most important concepts and facts from document content using Groq.

    Args:
        raw_text:   Full extracted document text
        api_key:    Groq API key
        num_points: Number of key points to extract

    Returns:
        Formatted key points string
    """
    os.environ["GROQ_API_KEY"] = api_key

    text_excerpt = raw_text[:8000] if len(raw_text) > 8000 else raw_text

    prompt = f"""You are an expert academic analyst.
Extract exactly {num_points} of the most important key points from the document below.

Format your response as:

🔑 KEY POINTS FROM THE DOCUMENT
================================

1. **[Topic/Concept Name]**
   [Clear, concise explanation in 1-3 sentences]

2. **[Topic/Concept Name]**
   [Clear, concise explanation in 1-3 sentences]

(continue for all {num_points} points)

---

IMPORTANT DEFINITIONS & TERMS:
[List any critical terms or definitions mentioned]

---

CORE TAKEAWAYS:
[2-3 sentence overall conclusion]

Rules:
- Extract only what is explicitly in the document.
- Prioritize concepts that appear most important.
- Use plain, clear language.

Document Content:
{text_excerpt}

Key Points:"""

    llm = ChatGroq(
        model=GROQ_MODEL,
        groq_api_key=api_key,
        temperature=0.2,
        max_tokens=2000,
    )
    response = llm.invoke(prompt)
    return response.content
