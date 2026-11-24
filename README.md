# Document Question Answering System using Retrieval-Augmented Generation (RAG)

An academic internship project built for **B.Tech CSE (AI & ML)** demonstrating a complete, working **Retrieval-Augmented Generation (RAG)** pipeline using **Python, LangChain, ChromaDB, and HuggingFace Embeddings**.

The system ingests the official **NIST AI 600-1** specification (*"Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile"*, 64 pages, published July 2024), breaks it into overlapping chunks, stores embeddings in a persistent vector database, and answers user questions with verified page citations while preventing hallucinations.

---

## 📌 Overview

Large Language Models (LLMs) are great at answering general questions, but they have two big limitations:
1. **They don't know your specific documents:** Models have a training cutoff and cannot see private, internal, or recently published PDFs.
2. **They hallucinate:** If an LLM doesn't know an answer, it often generates convincing but incorrect facts.

**Retrieval-Augmented Generation (RAG)** solves this problem by:
1. Searching a local vector database for the most relevant paragraphs from the document.
2. Passing those paragraphs into the prompt as factual context.
3. Instructing the model to answer **only** using the retrieved passages and cite the page numbers.

---

## ✨ Features

- **Real-World Technical Dataset:** Ingests the 64-page official NIST AI 600-1 Generative AI Profile.
- **Recursive Character Chunking:** Splits text into ~600 character passages with 100 character overlap to preserve sentence boundaries.
- **100% Free & Local Embeddings:** Uses `all-MiniLM-L6-v2` (384 dimensions) running locally via ONNX Runtime without needing paid API keys.
- **Persistent Vector Storage:** Stores vectors on disk in ChromaDB (`data/chroma_db/`) so you only embed the document once.
- **Semantic Retrieval:** Calculates Cosine Similarity to fetch the top 3 most relevant passages for any question.
- **Anti-Hallucination Guardrail:** If a question is not covered in the PDF, the system strictly outputs: *"The provided document does not contain this information."*
- **A/B Benchmark Comparison:** Built-in evaluation comparing **Without RAG vs. With RAG** on 5 specific test cases.
- **Interactive Streamlit Web UI:** Clean, simple graphical interface with expandable source citations.

---

## 🛠️ Technologies Used

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core programming language |
| **Document Loader** | LangChain `PyPDFLoader` | Reads text and page metadata from PDFs |
| **Text Splitter** | LangChain `RecursiveCharacterTextSplitter` | Breaks pages into overlapping chunks |
| **Embedding Model** | HuggingFace `all-MiniLM-L6-v2` | Generates 384-dimensional dense vectors locally |
| **Vector Database** | ChromaDB (PersistentClient) | Stores vectors and runs HNSW Cosine Similarity search |
| **Web Interface** | Streamlit | Simple interactive web application for live demos |
| **Testing** | Pytest | Automated unit tests for chunking and retrieval |

---

## 📂 Project Structure

```text
3.RAG System/
├── data/
│   ├── raw/
│   │   └── NIST.AI.600-1.pdf          <- Source PDF document (64 Pages)
│   └── chroma_db/                     <- Persistent ChromaDB vector store
├── src/
│   ├── __init__.py                    <- Package identifier
│   ├── config.py                      <- Central paths and hyperparameters
│   ├── loader.py                      <- Loads PDF and cleans whitespace
│   ├── chunker.py                     <- Recursive text splitter (600 chars / 100 overlap)
│   ├── embeddings.py                  <- Loads local all-MiniLM-L6-v2 embeddings
│   ├── vectorstore.py                 <- ChromaDB indexing and persistence
│   ├── retriever.py                   <- Cosine similarity search (top-k retrieval)
│   └── generator.py                   <- Prompt injection and grounded answer generation
├── tests/
│   ├── __init__.py
│   ├── test_chunking.py               <- Tests document loading and chunk bounds
│   └── test_retrieval.py              <- Tests vector store and similarity search
├── app.py                             <- Streamlit web application
├── benchmark_compare.py               <- Benchmark script: With RAG vs Without RAG
├── requirements.txt                   <- Project dependencies
├── .env.example                       <- Environment variables template
├── .gitignore                         <- Excludes .env and cache files
└── README.md                          <- Project documentation
```

---

## 🧠 How RAG Works in This Project

```text
  [ NIST.AI.600-1.pdf ] (64 Pages)
           │
           ▼
     src/loader.py       -> Extracts text page by page with metadata
           │
           ▼
     src/chunker.py      -> Creates 344 chunks (600 chars, 100 overlap)
           │
           ▼
   src/embeddings.py     -> Converts text into 384-d numbers (all-MiniLM-L6-v2)
           │
           ▼
   src/vectorstore.py    -> Saves vectors to disk in data/chroma_db/
           │
  ═════════╪═════════════════════════════════════════════════════════════════
           │  (When User Asks a Question)
           ▼
   src/retriever.py      -> Finds top 3 closest chunks using Cosine Similarity
           │
           ▼
   src/generator.py      -> Injects chunks into prompt template:
                            "Answer ONLY using the provided context..."
           │
           ▼
      Final Answer with Verified Page Citations
```

---

## 💻 Installation & Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/pallasandeepkumar125029-hub/rag-document-qa-system.git
cd rag-document-qa-system
```

### Step 2: Create a Virtual Environment
```bash
# Windows:
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify with Unit Tests
```bash
python -m pytest tests/
```
*(All 4 unit tests should pass in under 8 seconds).*

---

## 🚀 Usage

### 1. Run the Streamlit Web Application (Recommended for Demos)
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

<!-- SCREENSHOT PLACEHOLDER 1 -->
```text
+-----------------------------------------------------------------------------------+
| [SCREENSHOT PLACEHOLDER]                                                          |
| Streamlit Web UI showing Question Input, Final Answer, and Retrieved Chunks       |
| Add screenshot to: docs/screenshots/streamlit_ui.png                              |
+-----------------------------------------------------------------------------------+
```

### 2. Run the A/B Benchmark (With vs. Without RAG)
```bash
python benchmark_compare.py
```

### 3. Test Individual Modules from Terminal
```bash
# Test PDF loading:
python -m src.loader

# Test text chunking:
python -m src.chunker

# Test vector indexing:
python -m src.vectorstore

# Test semantic retrieval:
python -m src.retriever

# Test complete RAG generation:
python -m src.generator
```

---

## 📊 With RAG vs. Without RAG Results

Here is the actual comparison generated by `benchmark_compare.py` across 5 test cases from `NIST.AI.600-1.pdf`:

| ID | Question Topic | Without RAG (Generic / Parametric Memory) | With RAG (Our Grounded Pipeline) | Source Page |
| :---: | :--- | :--- | :--- | :---: |
| **1** | **Action ID GV-4.1-001** | ❌ **Vague / Hallucinated:** Does not know what "GV-4.1-001" is; gives generic AI explainability advice. | ✅ **Exact Fact:** Recommends gradient-based attributions, occlusion/term reduction, counterfactual prompts, and analysis of embeddings. | **Page 21** |
| **2** | **Algorithmic Monoculture** | ⚠️ **Generic Guess:** Talks vaguely about diversity in software algorithms. | ✅ **Exact Policy Definition:** Cites repeated use of the same model in consequential decision-making (employment/lending) causing correlated failures. | **Page 58 (Footnote 3)** |
| **3** | **Confabulation Definition** | ⚠️ **Medical Definition:** Defines confabulation as a human psychiatric memory disorder. | ✅ **Exact NIST Standard:** Defines production of confidently stated but false content, noting footnote 6 that "hallucination" anthropomorphizes AI. | **Page 10** |
| **4** | **Action ID GV-1.2-001** | ❌ **Vague Refusal:** States it has no access to specific clause numbers. | ✅ **Exact Fact:** Outlines establishing transparency policies for origin and history of training data while balancing proprietary interests. | **Page 18** |
| **5** | **Vitamin C Dosage (Trap)** | ❌ **Hallucinated Medical Advice:** Prescribes 65–90 mg/day from pre-trained training data. | ✅ **Strict Refusal:** *"The provided document does not contain this information."* (Zero hallucination). | **N/A** |

<!-- SCREENSHOT PLACEHOLDER 2 -->
```text
+-----------------------------------------------------------------------------------+
| [SCREENSHOT PLACEHOLDER]                                                          |
| Terminal output showing benchmark_compare.py execution and summary table          |
| Add screenshot to: docs/screenshots/benchmark_table.png                          |
+-----------------------------------------------------------------------------------+
```

---

## ❓ Sample Questions to Try

1. **"What is confabulation in Generative AI?"**  
   *(Retrieves Page 10, Section 2.2)*
2. **"What specific action does Action ID GV-4.1-001 recommend?"**  
   *(Retrieves Page 21, Section GOVERN 4.1)*
3. **"How does NIST AI 600-1 define algorithmic monoculture?"**  
   *(Retrieves Page 58, Footnote 3)*
4. **"What does Action ID GV-1.2-001 establish regarding data transparency?"**  
   *(Retrieves Page 18, Section GOVERN 1.2)*
5. **"What is the recommended dosage of aspirin?"**  
   *(Correctly triggers the negative constraint refusal)*

---

## 🔮 Future Improvements

- [ ] **Hybrid Search:** Combine BM25 keyword search with dense vector search to improve acronym retrieval.
- [ ] **Reranker Integration:** Add a lightweight cross-encoder reranker (e.g., `bge-reranker-base`) to re-score the top-k chunks.
- [ ] **Multi-Document Support:** Support querying across multiple different PDF files in `data/raw/` with a drop-down selector.
- [ ] **Chat History:** Add multi-turn conversational memory so users can ask follow-up questions.

---

## 👨‍💻 Author

- **Student Name:** Sandeep Kumar
- **Degree:** B.Tech in Computer Science & Engineering (AI & ML)
- **Role:** AI / Machine Learning Intern
- **GitHub:** [@pallasandeepkumar125029-hub](https://github.com/pallasandeepkumar125029-hub)
- **Project Title:** Document Question Answering System using RAG
- **Tools:** Python, LangChain, ChromaDB, HuggingFace, Streamlit

---

## 📜 License

This project is created for academic and educational purposes under the [MIT License](LICENSE).
