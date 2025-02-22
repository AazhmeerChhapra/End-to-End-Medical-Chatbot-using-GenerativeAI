from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings

def load_pdf(data):
    load = DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)
    document = load.load()
    return document
def text_split(document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    return text_splitter.split_documents(document)

def download_embeddings():
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

