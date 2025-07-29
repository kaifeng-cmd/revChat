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
from langchain.docstore.document import Document

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
    provider="fireworks-ai",
    api_key=os.environ.get("HF_TOKEN"),
    #model="google/gemma-3-27b-it"
    #model="Qwen/Qwen2.5-14B-Instruct"
    #model="meta-llama/Llama-3.3-70B-Instruct"
    model="meta-llama/Llama-3.1-8B-Instruct"
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

# Q&A parser
def parse_qna_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    chunks = []
    current_question = None
    current_answer_lines = []
    qid = 1

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("Question:"):
            # save previous pair
            if current_question and current_answer_lines:
                content = f"{current_question}\nAnswer: " + "\n".join(current_answer_lines).strip()
                chunks.append({
                    "content": content,
                    "metadata": {"id": f"q{qid}"}
                })
                qid += 1
                current_answer_lines = []

            current_question = stripped  # keep original Question: ... line

        elif stripped.startswith("Answer:"):
            current_answer_lines = [stripped[len("Answer:"):].strip()]
        else:
            if current_question:
                current_answer_lines.append(line.rstrip())

    # last pair
    if current_question and current_answer_lines:
        content = f"{current_question}\nAnswer: " + "\n".join(current_answer_lines).strip()
        chunks.append({
            "content": content,
            "metadata": {"id": f"q{qid}"}
        })

    return chunks

# Initialize or load Chroma collection
def initialize_vectorstore(doc_path: str, collection_name: str = "rev_collection") -> Chroma:
    persist_directory = "./app/data/chroma"
    client = chromadb.PersistentClient(path=persist_directory)
    try:
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

    print(f"Loading Q&A pairs from {doc_path}...")
    qa_chunks = parse_qna_file(doc_path)
    docs = [Document(page_content=chunk["content"], metadata=chunk["metadata"]) for chunk in qa_chunks]
    print("Docs to be stored in Chroma:")
    for doc in docs:
        print(f"ID: {doc.metadata['id']}\n{doc.page_content}\n{'-'*40}")
    print(f"Creating new vectorstore '{collection_name}'...")
    vectorstore = Chroma.from_documents(
        documents=docs,
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
        template="""Berdasarkan maklumat berikut (jangan jawab jika maklumat tiada): {context}
        \nSoalan: {question}
        \nJawapan: 
        \nKalau maklumat ada link, juga bagi link tersebut.
        """
    )
    
    def retrieve_context(question: str) -> str:
        results = vectorstore.similarity_search(question, k=2)
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
    #question = "I nak batalkan langganan bulanan TontonUp saya, macam mana?"
    question = "Knp i xboleh tengok selepas buat bayaran?"
    answer = rag_chain.invoke({"question": question})
    print(f"Soalan: {question}\nJawapan: {answer}")

if __name__ == "__main__":
    main()
