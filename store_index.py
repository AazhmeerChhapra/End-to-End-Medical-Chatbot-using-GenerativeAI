from src.helper import load_pdf, text_split, download_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import pinecone
import os

load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)


data = load_pdf('F:\End-to-End-Medical-Chatbot-using-GenerativeAI\data')
chunks = text_split(data)
embeddings = download_embeddings()

index = pc.Index('medical-chatbot')
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

from uuid import uuid4

from langchain_core.documents import Document
documnents = []

for t in chunks:
    temp = Document(page_content=t.page_content)
    documnents.append(temp)
uuids = [str(uuid4()) for _ in range(len(documnents))]


vector_store.add_documents(documents=documnents, ids=uuids)





