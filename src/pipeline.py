import os
from typing import List, Tuple, Any

from langchain_community.document_loaders import UnstructuredFileLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# -----------------------------
# Document loading and chunking
# -----------------------------

def load_and_chunk_document(
    file_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 60,
) -> List[Any]:
    """
    Load a document with Unstructured (fallback to simple text loader) and split into chunks.
    Returns a list of LangChain Document objects.
    """
    docs: List[Any] = []
    try:
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load()
    except Exception:
        # Fallback to basic text loader if unstructured has extra system deps
        loader = TextLoader(file_path)
        docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    return chunks


# -----------------------------
# Embeddings and Vector Store
# -----------------------------

def get_embeddings(
    model_name: str = "BAAI/bge-small-en-v1.5",
    device: str = "cpu",
) -> HuggingFaceEmbeddings:
    """Return a HuggingFaceEmbeddings instance configured for bge-small-en-v1.5."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_faiss_vectorstore(
    docs: List[Any],
    embeddings: HuggingFaceEmbeddings,
) -> Tuple[FAISS, Any]:
    """
    Build a FAISS vector store from documents and return (vectorstore, retriever).
    """
    vs = FAISS.from_documents(docs, embeddings)
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    return vs, retriever


# -----------------------------
# LLM and RAG Chain
# -----------------------------

def get_default_llm(
    model_name: str = "llama3-8b-8192",
    temperature: float = 0.0,
) -> ChatGroq:
    """Construct a Groq Llama3 chat model using GROQ_API_KEY from environment."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Please export GROQ_API_KEY before running."
        )
    return ChatGroq(model_name=model_name, temperature=temperature, groq_api_key=api_key)


def make_rag_chain(retriever: Any, llm: ChatGroq):
    """
    Create a simple RAG chain that retrieves, stuffs context, and answers concisely.
    Input: {"question": str}
    Output: str (final answer)
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer the question using ONLY the provided context.\n"
                "- If the answer is not in the context, say: 'I don't know based on the provided context.'\n"
                "- Be concise and strictly factual.\n"
                "Context:\n{context}",
            ),
            ("human", "Question: {question}"),
        ]
    )

    format_docs = RunnableLambda(lambda docs: "\n\n".join(d.page_content for d in docs))

    chain = {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    } | prompt | llm | StrOutputParser()

    return chain


# -----------------------------
# Helpers for evaluation
# -----------------------------

def get_contexts_for_questions(
    vectorstore: FAISS,
    questions: List[str],
    k: int = 4,
) -> List[List[str]]:
    """Return top-k context strings per question using vectorstore similarity search."""
    contexts: List[List[str]] = []
    for q in questions:
        docs = vectorstore.similarity_search(q, k=k)
        contexts.append([d.page_content for d in docs])
    return contexts
