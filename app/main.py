import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableLambda
from langchain_core.embeddings import Embeddings
from huggingface_hub import InferenceClient
from pathlib import Path
from typing import List
import chromadb

# define embedding class
class CustomHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=os.environ.get("HF_TOKEN"),
            model=model_name
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.client.feature_extraction(texts)
        return [e.tolist() if hasattr(e, 'tolist') else e for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        embedding = self.client.feature_extraction(text)
        return embedding.tolist() if hasattr(embedding, 'tolist') else embedding


embedding_function = CustomHuggingFaceEmbeddings()

# Initialize inference client
inference_client = InferenceClient(
    provider="featherless-ai",
    api_key=os.environ.get("HF_TOKEN"),
    #model="google/gemma-3-27b-it"
    model="Qwen/Qwen2.5-14B-Instruct"
)

# Load documents
def load_documents(doc_path: str) -> str:
    doc_path = Path(doc_path)
    text = ""
    if doc_path.suffix == ".txt":
        with open(doc_path, "r", encoding="utf-8") as f:
            text = f.read()
    elif not doc_path.exists():
        print(f"Error: Document not found at {doc_path}")
    else:
        print(f"Warning: Unsupported file type '{doc_path.suffix}'. Only '.txt' files are processed.")
    return text

# Split documents
def split_documents(text: str) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )
    return text_splitter.split_text(text)

# Initialize or load Chroma collection
def initialize_vectorstore(doc_path: str, collection_name: str = "rev_collection") -> Chroma:
    persist_directory = "./app/data/chroma"
    client = chromadb.PersistentClient(path=persist_directory)
    
    try:
        # Try to load existing collection
        vectorstore = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding_function,
        )
        if vectorstore._collection.count() > 0:
            print(f"Using existing collection: {collection_name}")
            return vectorstore
        else:
            print(f"Collection '{collection_name}' is empty, re-creating...")
    except Exception:
        print(f"Collection '{collection_name}' does not exist, creating new one...")
    
    # If collection does not exist or is empty, create new one
    print(f"Loading documents from {doc_path}...")
    text = load_documents(doc_path)
    if not text:
        raise ValueError(f"Failed to load text from {doc_path}. The file might be empty or in a wrong format.")

    print("Splitting documents...")
    chunks = split_documents(text)
    
    print(f"Creating new vectorstore '{collection_name}'...")
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_function,
        collection_name=collection_name,
        client=client
    )
    print("New vectorstore created and persisted.")
    return vectorstore

# LCEL RAG chain
def create_rag_chain(vectorstore: Chroma) -> RunnableSequence:
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="Berdasarkan maklumat berikut (jangan jawab jika maklumat tiada): {context}\nSoalan: {question}\nJawapan:"
    )
    
    def retrieve_context(question: str) -> str:
        results = vectorstore.similarity_search(question, k=3)
        return results[0].page_content if results else ""

    def generate_answer(inputs: dict) -> str:
        prompt = inputs["prompt"]
        # Compatible with StringPromptValue type
        if hasattr(prompt, "to_string"):
            prompt = prompt.to_string()
        else:
            prompt = str(prompt)
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer in Bahasa Malaysia if the question is in Malay."},
            {"role": "user", "content": prompt}
        ]
        completion = inference_client.chat.completions.create(
            messages=messages
        )
        return completion.choices[0].message.content

    chain = (
        {
            "context": lambda x: retrieve_context(x["question"]),
            "question": lambda x: x["question"],
        }
        | RunnableParallel(prompt=prompt_template)
        | RunnableLambda(generate_answer)
    )
    return chain

def main():
    doc_path = "app/documents/FAQ.txt"

    # for Chroma
    vectorstore = initialize_vectorstore(doc_path)

    # for RAG
    rag_chain = create_rag_chain(vectorstore)
    
    # User query
    question = "Bagaimana saya nak membatalkan langganan bulanan TontonUp saya?"
    answer = rag_chain.invoke({"question": question})
    print(f"Soalan: {question}\nJawapan: {answer}")

if __name__ == "__main__":
    main()
