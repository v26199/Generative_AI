from langchain.llms import CTransformers
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langchain.chains import create_retrieval_chain
from flask import Flask, render_template, jsonify, request
import os
from dotenv import load_dotenv


load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")

print("=====>")
print("PINECONE_INDEX_NAME: ", PINECONE_INDEX_NAME)

embeddings = download_hugging_face_embeddings()

docsearch = PineconeVectorStore.from_existing_index(PINECONE_INDEX_NAME, embeddings)

# use the model to get accurate answer most relavent.
llm = CTransformers(model="/Users/vishalpatel/Desktop/openaI/MediMate/model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})

retriever = docsearch.as_retriever(search_kwargs={'k': 2})

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt
)

retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

prefix_length = len('\nSystem: ')


app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = retrieval_chain.invoke({"input": input})
    print("\n\nResult===> ", result)
    answer = str(result['answer'][prefix_length:])
    print("\n\nAnswer===> ", answer)
    return answer



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)