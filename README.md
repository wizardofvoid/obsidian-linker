# Obsidian Linker

Obsidian Linker is an intelligent, agentic background tool powered by **LangGraph** and **LangChain** that automatically organizes your Obsidian vault. It reads your markdown notes, extracts core concepts, generates global tags, and seamlessly wires your knowledge graph together by injecting semantic cross-note links (`[[Related Note]]`).

## ✨ Key Features

- **Automated Concept Extraction:** Uses LLMs to deeply read and summarize the core ideas inside every markdown note.
- **Smart Tagging:** Automatically generates and appends relevant, global Obsidian-style hashtags (e.g., `#machine-learning`) to your notes.
- **Semantic Cross-Linking (FAISS):** Embeds all extracted concepts using Google's Generative AI embeddings and uses a local FAISS vectorstore to find hidden, semantic connections between different notes in your vault.
- **Strict Quality Control:** Uses a verification LLM and structural filtering to guarantee that proposed links are meaningful, preventing hallucinations and self-linking.
- **Cost-Efficient Caching:** Calculates MD5 hashes of your notes (ignoring the auto-generated tags/links) so it only burns API tokens on files that you have actually edited or created since the last run.
- **API Key Rotation:** Built-in exponential backoff and automatic key rotation for both Groq and Google APIs to bypass free-tier rate limits (`429 RESOURCE_EXHAUSTED`).

## 🛠️ Tech Stack

- **Python 3.x**
- **[LangGraph](https://python.langchain.com/docs/langgraph) / LangChain**: Core orchestration and LLM pipelines.
- **[Groq](https://groq.com/)**: Blazing fast inference for Extraction and Verification tasks (default: `llama-3.3-70b-versatile`).
- **Google Generative AI**: Semantic embeddings (`models/gemini-embedding-2`).
- **FAISS**: Local vector database for semantic similarity search.

## 🚀 Setup & Installation

1. **Clone the repository.**
2. **Set up the Conda Environment:**
   Ensure you are using the designated conda environment.
   ```bash
   conda activate langgraph
   ```
3. **Install Dependencies:**
   Ensure you have LangChain, LangGraph, FAISS, Groq, and Google Generative AI packages installed.
4. **Configure Environment Variables:**
   Create a `.env` file in the root of the project. You can provide multiple API keys to take advantage of the automatic rotation feature!
   ```env
   # Groq Keys (Extraction & Verification)
   GROQ_API_KEY_1="gsk_your_groq_key_here"
   GROQ_API_KEY_2="gsk_your_second_groq_key_here"

   # Google Keys (Embeddings)
   GOOGLE_API_KEY_1="AIza_your_google_key_here"
   GOOGLE_API_KEY_2="AIza_your_second_google_key_here"
   
   # Optional overrides
   LLM_MODEL="llama-3.3-70b-versatile"
   ```

## 🧠 How It Works

1. **Vault Reader (`nodes/io.py`)**: Scans your vault directory for `.md` files and compares them against the `.linker_cache.json` hash cache to determine which notes are new or modified.
2. **Concept Extractor (`nodes/concepts.py`)**: Batches the new notes and sends them to Groq to extract structured JSON concepts and tags.
3. **Concept Verifier**: Cleans and verifies the extracted concepts/tags and injects them into the local FAISS index (`.linker_faiss_index`).
4. **Relationship Extractor (`nodes/relations.py`)**: Queries the FAISS index to find semantically similar historical concepts, then asks the LLM to identify meaningful cross-note relationships.
5. **Relationship Verifier**: Double-checks the proposed relationships and strictly filters out structural invalidities (like linking a note to itself or to a non-existent note).
6. **Link Writer (`nodes/io.py`)**: Safely appends the `## Tags` and `## Related Links` sections directly into your local `.md` files.

## 🖥️ Usage

Run the main pipeline:
```bash
python .\main.py
```
*(If you aren't already inside the conda environment, use `conda run -n langgraph python .\main.py`)*

The script will prompt you for the absolute path to your Obsidian vault (e.g., `C:\Users\saraf\Documents\VOID`). It will then run asynchronously and summarize how many tags and links were created.
