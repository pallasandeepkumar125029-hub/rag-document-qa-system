"""
app.py
------
Streamlit Web Application for B.Tech RAG Internship Project.
Allows users to ask questions on NIST.AI.600-1.pdf, view retrieved chunks,
check similarity scores, and read the grounded answer with citations.
"""

import streamlit as st
from src.generator import generate_rag_answer

# -----------------------------------------------------------------------------
# Page Configuration & Simple Header
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Document QA System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Question Answering System (RAG)")
st.write("**B.Tech CSE (AI & ML) Internship Project** | Tech Stack: Python, LangChain, ChromaDB, HuggingFace")
st.write("Source Dataset: **NIST AI 600-1 Generative AI Profile** (64 Pages)")
st.markdown("---")

# -----------------------------------------------------------------------------
# Sidebar: Project Info & Sample Questions
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Project Details")
    st.write("- **Vector Database:** ChromaDB")
    st.write("- **Embedding Model:** all-MiniLM-L6-v2 (384 dims)")
    st.write("- **Total Chunks:** 344 Chunks (600 chars, 100 overlap)")
    st.write("- **Persistence:** Disk (`data/chroma_db/`)")
    st.markdown("---")
    
    st.subheader("💡 Try Sample Questions:")
    if st.button("1. What is confabulation?"):
        st.session_state["sample_query"] = "What is confabulation?"
    if st.button("2. Action ID GV-4.1-001 Explainability"):
        st.session_state["sample_query"] = "What specific action does Action ID GV-4.1-001 recommend to address risks associated with a lack of explainability in GAI systems?"
    if st.button("3. Algorithmic Monoculture Definition"):
        st.session_state["sample_query"] = "How does NIST AI 600-1 define algorithmic monoculture and what risk does it introduce?"
    if st.button("4. Vitamin C (Hallucination Test)"):
        st.session_state["sample_query"] = "What is the recommended daily dosage of Vitamin C according to NIST AI 600-1?"

# -----------------------------------------------------------------------------
# Section 1: User Question Input
# -----------------------------------------------------------------------------
default_text = st.session_state.get("sample_query", "")
user_question = st.text_input(
    "1. Enter your question about the document:",
    value=default_text,
    placeholder="e.g., What is confabulation in Generative AI?"
)

submit_button = st.button("🔍 Submit Question", type="primary")

if submit_button:
    if not user_question.strip():
        st.warning("Please enter a question first!")
    else:
        with st.spinner("Searching ChromaDB and generating answer..."):
            # Call generator function (which calls retriever.py and ChromaDB)
            result = generate_rag_answer(user_question, top_k=3)

        # ---------------------------------------------------------------------
        # Section 4: Final Answer (Prominently displayed)
        # ---------------------------------------------------------------------
        st.subheader("4. Final Answer")
        if result["source_pages"]:
            st.success(result["answer"])
        else:
            # Negative constraint message if question not in document
            st.info(result["answer"])

        # ---------------------------------------------------------------------
        # Section 5: Source Citations
        # ---------------------------------------------------------------------
        st.subheader("5. Source Citations")
        if result["source_pages"]:
            pages_list = ", ".join(f"Page {p}" for p in result["source_pages"])
            st.markdown(f"Verified Source: **`NIST.AI.600-1.pdf`** ({pages_list})")
        else:
            st.markdown("Verified Source: **None** (Information not present in document)")

        st.markdown("---")

        # ---------------------------------------------------------------------
        # Section 2 & 3: Retrieved Chunks & Similarity Scores
        # ---------------------------------------------------------------------
        st.subheader(f"2 & 3. Retrieved Chunks & Similarity Scores ({len(result['chunks'])} Chunks)")
        st.caption("Passages retrieved from ChromaDB using Cosine Similarity:")

        if result["chunks"]:
            for i, chunk in enumerate(result["chunks"], start=1):
                score = chunk["similarity_score"]
                page = chunk["page"]
                source = chunk["source"]
                chunk_id = chunk["chunk_id"]

                with st.expander(f"Chunk #{i} — Similarity Score: {score} | Page {page}"):
                    st.write(f"**Source Document:** `{source}` | **Page:** {page} | **ID:** `{chunk_id}`")
                    st.write(f"**Cosine Similarity Score:** `{score}` (Range 0 to 1, higher = closer)")
                    st.text_area("Passage Text:", value=chunk["content"], height=120, key=f"chunk_text_{i}", disabled=True)
        else:
            st.write("No relevant chunks found above similarity threshold.")
