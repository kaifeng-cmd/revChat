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
import re
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Self-defined embedding class
# becuz I don't directly download the embedding model from huggingface
# or use paid service provider embedding model
# I use the huggingface inference client to get the embedding
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
    # provider="featherless-ai",
    provider="fireworks-ai",
    api_key=os.environ.get("HF_TOKEN"),
    #model="google/gemma-3-27b-it"
    #model="Qwen/Qwen2.5-14B-Instruct"
    #model="Qwen/Qwen2.5-1.5B-Instruct"
    #model="meta-llama/Llama-3.3-70B-Instruct"
    model="meta-llama/Llama-3.1-8B-Instruct"
)

# Q&A parser
def parse_qna_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split the text before each line that starts with "Question:"
    raw_chunks = re.split(r'(?m)(^Question:.*)', content)

    chunks = []

    for i in range(1, len(raw_chunks), 2):
        # Combine the "Question:..." line with the following answer
        qna_pair = (raw_chunks[i] + raw_chunks[i+1]).strip()
        if qna_pair:
            chunks.append({
                "content": qna_pair,
                "metadata": {"id": f"q{len(chunks) + 1}"}
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

# Streamlit chat UI
def streamlit_chat():
    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
    st.title("RAG Chatbot Demo")

    doc_path = "app/documents/FAQ.txt"
    vectorstore = initialize_vectorstore(doc_path)
    rag_chain = create_rag_chain(vectorstore)

    # Streamlit chat interface
    st.write("Hi, sila tanya soalan anda di bawah:")
    user_input = st.chat_input("Taip soalan anda...")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            answer = rag_chain.invoke({"question": user_input})
            st.markdown(answer)

    """
    Contoh soalan:
    \nI nak batalkan langganan bulanan TontonUp saya, macam mana?
    \nKnp i xboleh tengok selepas buat bayaran?
    """

#ignore, for my CLI testing only
# def main():
#     doc_path = "app/documents/FAQ.txt"
#     vectorstore = initialize_vectorstore(doc_path)
#     rag_chain = create_rag_chain(vectorstore)
#     #question = "I nak batalkan langganan bulanan TontonUp saya, macam mana?"
#     #question = "Knp i xboleh tengok selepas buat bayaran?"
#     question = "Bagaimana saya nak membatalkan langganan bulanan TontonUp saya?"
#     answer = rag_chain.invoke({"question": question})
#     print(f"Soalan: {question}\nJawapan: {answer}")

if __name__ == "__main__":
    streamlit_chat()
