from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_nomic.embeddings import NomicEmbeddings  
from langchain_ollama import ChatOllama

# LLM setup
llm = "llama3.1:8b"
llm = ChatOllama(model=llm, temperature=0)

