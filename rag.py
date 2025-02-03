from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_nomic.embeddings import NomicEmbeddings  
from langchain_ollama import ChatOllama

# LLM setup
model = "llama3.1:8b"
llm = ChatOllama(model=model, temperature=0)
llm_json_mode = ChatOllama(model=model, temperature=0, output_format="json")

# embeddings model setup
embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")

# documents

files = [
  "files/energy-surplus.pdf",
  "files/nutrients.pdf",
  "files/optimal-training.pdf",
  "files/supplementation.pdf",
  "files/timing.pdf"
]

# load documents
docs = []
for file in files:
  loader = PyPDFLoader(file)
  docs += loader.load()

# split documents
text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=512, chunk_overlap=100, length_function=len
)

doc_splits = text_splitter.split_documents(docs)

# add to vector database
vectorstore = Chroma.from_documents(
  documents=doc_splits,
  embedding=embeddings
)

# create retriever
k = min(3, len(doc_splits)) # ensure k does not exceed available chunks
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": min(5, len(doc_splits)), "score_threshold": 0.05},  # Lower threshold, increase retrieved docs
)
