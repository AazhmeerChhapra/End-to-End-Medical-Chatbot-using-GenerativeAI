from src.helper import download_embeddings
from flask import Flask, jsonify, render_template, request
from langchain.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from src.prompt import *
import os
from dotenv import load_dotenv
app = Flask(__name__, template_folder='template')
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
embeddings = download_embeddings()

index = 'medical-chatbot'

docsearch = Pinecone.from_existing_index(index, embeddings)
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

chain_type_kwargs = {'prompt':prompt}

llm = CTransformers(model='F:\End-to-End-Medical-Chatbot-using-GenerativeAI\model\llama-2-7b-chat.ggmlv3.q4_0.bin',
                    model_type = 'llama',
                    config = {
                        'max_new_tokens':512,
                        'temperature':0.8
                    })
qa = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = "stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents = True,
    chain_type_kwargs = chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods = ["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa({'query':input})
    print("Respinse: ", result['result'])
    return str(result['result'])
if __name__=='__main__':
    app.run(host="0.0.0.0", port=8080, debug = True)