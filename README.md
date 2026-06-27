🤖 Hugging Face RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, Sentence Transformers, FAISS, and the Hugging Face Datasets library. The application retrieves the most relevant information from a knowledge base and uses an LLM to generate context-aware responses.

⸻

📌 Overview

This project demonstrates a complete Retrieval-Augmented Generation (RAG) pipeline:

* Load a Hugging Face dataset as the knowledge source
* Split the dataset into meaningful chunks
* Generate semantic embeddings
* Store embeddings in a FAISS vector database
* Retrieve the most relevant chunks for a user query
* Generate an answer using an LLM
* Display the retrieved sources for transparency

Unlike a traditional chatbot, this application grounds its responses using retrieved documents, reducing hallucinations and improving factual accuracy.

⸻

✨ Features

* 📚 Retrieval-Augmented Generation (RAG)
* 🤗 Uses Hugging Face datasets
* 🔍 Semantic similarity search with FAISS
* 🧠 Sentence Transformer embeddings
* 💬 Streamlit chat interface
* 📂 Expandable retrieved source viewer
* 🔄 Context deduplication
* 🎯 Improved prompt engineering
* ⚡ Fast local inference
* 🌙 Clean dark-themed UI

⸻

🏗️ Project Architecture

User Query
     │
     ▼
Retrieve Top-K Relevant Chunks
     │
     ▼
Remove Duplicate Contexts
     │
     ▼
Construct Prompt
     │
     ▼
LLM Generation
     │
     ▼
Generated Answer + Retrieved Sources

⸻

📁 Project Structure

hugging-face-rag-chatbot/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── your_document.pdf
│
├── utils/
│   ├── loader.py
│   ├── embedder.py
│   └── retriever.py
│
└── screenshots/
    └── demo.png

⸻

📄 Knowledge Source

The chatbot uses a Hugging Face dataset containing cyber crime complaint records as its knowledge base.

The dataset includes information such as:

* Crime Category
* Sub Category
* Complaint Details
* Additional Information

The dataset is transformed into searchable text chunks before indexing.

⸻

✂️ How Chunking Works

The loader converts each dataset record into a structured text representation.

Example:

Category: Any Other Cyber Crime
Sub Category: Other Crime
Additional Info:
OnePlus phone stolen from IFFCO Chowk Bus Stand...

Each record becomes one semantic chunk.

This preserves contextual meaning while allowing efficient retrieval.

⸻

🧠 Embedding Model

The application generates embeddings using:

sentence-transformers

Recommended model:

all-MiniLM-L6-v2

This lightweight embedding model provides:

* Fast inference
* Good semantic understanding
* Low memory usage
* Excellent retrieval quality

⸻

🗄️ Vector Store

The generated embeddings are stored in:

FAISS (Facebook AI Similarity Search)

FAISS enables fast nearest-neighbor search even for thousands of documents.

⸻

🔍 Retrieval Pipeline

1. User enters a question.
2. The question is embedded.
3. FAISS retrieves the Top-K similar chunks.
4. Duplicate contexts are removed.
5. Clean context is passed to the LLM.
6. The generated response is displayed.
7. Retrieved source documents are shown for transparency.

⸻

🚀 Recent Improvements

Several improvements were added to enhance response quality:

✅ Context Deduplication

* Removes duplicate retrieved chunks
* Eliminates repetitive responses
* Improves prompt quality

⸻

✅ Better Prompt Engineering

Retrieved contexts are formatted as:

[Source 1]
...
[Source 2]
...

This helps the LLM better understand individual pieces of evidence.

⸻

✅ Expandable Source Viewer

Users can inspect the exact retrieved contexts using the View Retrieved Sources panel.

This improves explainability and transparency.

⸻

✅ Increased Retrieval Quality

* Retrieves more candidate chunks
* Filters duplicate contexts
* Uses the most informative sources

⸻

✅ Cleaner UI

* Chat interface
* Expandable retrieved sources
* Improved readability
* Better visual hierarchy

⸻

💻 Technologies Used

* Python
* Streamlit
* Hugging Face Datasets
* Sentence Transformers
* FAISS
* Transformers
* PyTorch

⸻

⚙️ Installation

Clone the repository:

git clone https://github.com/Revanth-arc/hugging-face-rag-chatbot.git
cd hugging-face-rag-chatbot

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

⸻

📸 Screenshot

Example of the application interface:

⸻

📖 Example Query

Give an overview of the cases in the dataset.

Example Response:

* Summary of retrieved cyber crime complaints
* Relevant categories
* Complaint details
* Supporting retrieved sources

⸻

🔮 Future Improvements

Given additional development time, the project could be extended with:

* Streaming LLM responses
* Multi-turn conversational memory
* Hybrid search (BM25 + Semantic Search)
* Metadata filtering
* Query rewriting
* Better chunk ranking
* Source citations
* PDF upload support
* Multiple document collections
* ChromaDB or LanceDB backend
* Authentication
* Deployment on Hugging Face Spaces

⸻

📷 Demo

The chatbot provides:

* Interactive chat interface
* Context-aware responses
* Semantic retrieval
* Transparent retrieved sources
* Clean and responsive UI

⸻

👨‍💻 Author

Revanth Chary

B.Tech Computer Science (Artificial Intelligence)

ICFAI University, Hyderabad

GitHub: https://github.com/Revanth-arc

⸻

📜 License

This project is developed for educational and internship purposes.
