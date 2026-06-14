import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    # print("PINECONE_API_KEY=", os.environ["PINECONE_API_KEY"])

    loader = TextLoader("mediumblog1.txt")
    doc = loader.load()

    print("splitting...")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(doc)

    print(f"created {len(texts)} chunks")

    embedding = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    vectorstore_from_docs = PineconeVectorStore.from_documents(
        texts,
        embedding,
        index_name=os.getenv("INDEX_NAME")
    )

    print(f"created vectorstore_from_docs")
