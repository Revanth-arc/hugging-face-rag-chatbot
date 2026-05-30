import streamlit as st
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="HF RAG Chatbot", page_icon="💬", layout="wide")
st.title("💬 Hugging Face RAG Chatbot")

st.write(
    "This chatbot loads text from Hugging Face, builds a vector index, "
    "and answers your questions using retrieved context."
)

# ---------------------------
# Cached models
# ---------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_generator():
    return pipeline("text2text-generation", model="google/flan-t5-small")

DATASETS_TO_LOAD = [
    {"name": "ag_news", "display": "AG News"},
    {"name": "crime_and_punish", "display": "Crime and Punishment"},
    {"name": "Dev523/Crime-Reports-Dataset", "display": "Crime Reports Dataset"},
    {"name": "dvilasuero/fraud-email-detection", "display": "Fraud Email Detection"},
    {"name": "mstz/financial_fraud_detection", "display": "Financial Fraud Detection"},
]

PREFERRED_TEXT_COLUMNS = [
    "text",
    "content",
    "report",
    "complaint",
    "description",
    "message",
    "body",
    "email_body",
    "subject",
    "email",
    "line",
]


def extract_text_from_row(row):
    if not isinstance(row, dict):
        row = dict(row)

    for key in PREFERRED_TEXT_COLUMNS:
        if key in row and row[key] is not None:
            value = row[key]
            if isinstance(value, list):
                value = " ".join(str(item) for item in value if item is not None)
            elif not isinstance(value, str):
                value = str(value)
            if value.strip():
                return value.strip()

    parts = []
    for key, value in row.items():
        if value is None:
            continue
        if isinstance(value, list):
            value = " ".join(str(item) for item in value if item is not None)
        else:
            value = str(value)
        if value.strip():
            parts.append(f"{key}: {value.strip()}")
    return "\n".join(parts).strip()


@st.cache_data
def load_hf_texts(limit=1000):
    texts = []
    dataset_stats = []
    warnings = []

    for dataset_meta in DATASETS_TO_LOAD:
        try:
            ds_dict = load_dataset(dataset_meta["name"])
            if isinstance(ds_dict, dict):
                if "train" in ds_dict:
                    dataset = ds_dict["train"]
                else:
                    first_split = list(ds_dict.keys())[0]
                    dataset = ds_dict[first_split]
            else:
                dataset = ds_dict

            loaded = 0
            for i in range(min(limit, len(dataset))):
                row = dataset[i]
                text = extract_text_from_row(row)
                if text:
                    texts.append(text)
                    loaded += 1
            dataset_stats.append((dataset_meta["display"], loaded))
        except Exception as err:
            warnings.append(f"Skipped {dataset_meta['display']}: {err}")

    return texts, dataset_stats, warnings

# ---------------------------
# Text chunking
# ---------------------------
def chunk_text(text, chunk_size=700, overlap=120):
    text = " ".join(text.split())
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + chunk_size])
        i += chunk_size - overlap
    return [c for c in chunks if c.strip()]

# ---------------------------
# Build vector index
# ---------------------------
def build_index(chunks):
    embedder = load_embedder()
    embeddings = embedder.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

# ---------------------------
# Deduplicate contexts
# ---------------------------
def deduplicate_contexts(contexts, similarity_threshold=0.85):
    """Remove near-duplicate contexts to avoid repetition."""
    if not contexts:
        return []
    
    unique_contexts = []
    seen_normalized = set()
    
    for context in contexts:
        # Normalize text for comparison
        normalized = " ".join(context.lower().split())
        
        # Skip if we've already seen an identical normalized version
        if normalized in seen_normalized:
            continue
        
        # Check similarity with existing contexts
        is_duplicate = False
        for existing in unique_contexts:
            existing_normalized = " ".join(existing.lower().split())
            
            # Calculate word-level similarity
            context_words = set(normalized.split())
            existing_words = set(existing_normalized.split())
            
            if len(context_words) > 0 and len(existing_words) > 0:
                common_words = len(context_words & existing_words)
                total_words = len(context_words | existing_words)
                similarity = common_words / total_words if total_words > 0 else 0
                
                if similarity > similarity_threshold:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            unique_contexts.append(context)
            seen_normalized.add(normalized)
    
    return unique_contexts[:8]  # Limit to top 8 unique contexts

# ---------------------------
# Retrieve relevant chunks
# ---------------------------
def retrieve(query, chunks, index, top_k=12):
    if index is None or not chunks:
        return []

    embedder = load_embedder()
    q = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, ids = index.search(q, top_k)

    results = []
    for idx in ids[0]:
        if idx != -1 and idx < len(chunks):
            results.append(chunks[idx])
    return results

# ---------------------------
# Make prompt
# ---------------------------
def make_prompt(question, contexts):
    """Create a well-formatted prompt with deduplicated context."""
    unique_contexts = deduplicate_contexts(contexts)
    
    if not unique_contexts:
        context_text = "No relevant context found."
    else:
        # Format contexts with clear source labels
        formatted_contexts = []
        for i, c in enumerate(unique_contexts, 1):
            # Clean up the context
            c_clean = " ".join(c.split())
            formatted_contexts.append(f"[Source {i}] {c_clean}")
        context_text = "\n\n".join(formatted_contexts)
    
    return f"""Based on the following context, provide a clear and concise answer to the question. 
If the context doesn't contain relevant information, say so.

Context:
{context_text}

Question: {question}

Answer:"""

# ---------------------------
# Session state
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("Build knowledge base")
build_clicked = st.sidebar.button("Load Hugging Face data and build index")

if build_clicked:
    try:
        with st.spinner("Loading data from Hugging Face..."):
            texts, dataset_stats, warnings = load_hf_texts()

        with st.spinner("Chunking text..."):
            chunks = []
            for text in texts:
                chunks.extend(chunk_text(text))

        with st.spinner("Building vector index..."):
            index = build_index(chunks)

        st.session_state.chunks = chunks
        st.session_state.index = index
        st.sidebar.success(f"Loaded {len(texts)} text entries and built {len(chunks)} chunks.")
        for name, count in dataset_stats:
            st.sidebar.write(f"- {name}: {count} entries")
        for warning in warnings:
            st.sidebar.warning(warning)
    except Exception as err:
        st.sidebar.error(f"Failed to build the knowledge base: {err}")

if st.sidebar.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()

# ---------------------------
# Show chat history
# ---------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------------------
# Chat input
# ---------------------------
question = st.chat_input("Ask a question about the Hugging Face data...")

if question and question.strip():
    question = question.strip()
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    if st.session_state.index is None:
        reply = "Please load the Hugging Face data first."
        with st.chat_message("assistant"):
            st.write(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
    else:
        with st.spinner("Thinking..."):
            contexts = retrieve(question, st.session_state.chunks, st.session_state.index)
            prompt = make_prompt(question, contexts)

            generator = load_generator()
            result = generator(prompt, max_new_tokens=200, do_sample=False)
            answer = result[0]["generated_text"]

        with st.chat_message("assistant"):
            st.write(answer)
            
            # Show deduplicated source context
            unique_contexts = deduplicate_contexts(contexts)
            if unique_contexts:
                with st.expander("📋 View Retrieved Sources", expanded=False):
                    for i, c in enumerate(unique_contexts, 1):
                        # Display source with truncation for very long contexts
                        display_text = c if len(c) <= 500 else c[:500] + "..."
                        st.markdown(f"**Source {i}:**")
                        st.text(display_text)
                        st.divider()

        st.session_state.messages.append({"role": "assistant", "content": answer})
