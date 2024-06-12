import streamlit as st
from streamlit_chat import message
import tempfile
import json
import fitz  # PyMuPDF for handling PDF files
from langchain_community.document_loaders import CSVLoader, JSONLoader, TextLoader  # Update imports
from langchain_community.embeddings import HuggingFaceEmbeddings  # Update imports
from langchain_community.vectorstores import FAISS  # Update imports
from langchain_community.llms import CTransformers  # Update imports
from langchain.chains import ConversationalRetrievalChain

DB_FAISS_PATH = 'vectorstore/db_faiss'

# Loading the model
def load_llm():
    llm = CTransformers(
        model="model/llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

# Function to load different file types
def load_file(file):
    file_type = file.name.split('.')[-1]
    if file_type == 'csv':
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name
        loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
        data = loader.load()
        
    elif file_type == 'json':
        json_data = json.loads(file.getvalue())
        data = [{"text": json.dumps(json_data)}]
        loader = JSONLoader(data)
        
    elif file_type == 'txt':
        text = file.read().decode('utf-8')
        data = [{"text": text}]
        loader = TextLoader(data)
        
    elif file_type == 'pdf':
        text = ""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name
        with fitz.open(tmp_file_path) as doc:
            for page in doc:
                text += page.get_text()
        data = [{"text": text}]
        
    else:
        st.error("Unsupported file type")
        return None
    
    return data

st.title("Chat Application For your custom Data")
uploaded_file = st.sidebar.file_uploader("Upload your Data", type=["csv", "json", "txt", "pdf"])

if uploaded_file:
    data = load_file(uploaded_file)
    
    if data:
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                           model_kwargs={'device': 'cpu'})

        db = FAISS.from_documents(data, embeddings)
        db.save_local(DB_FAISS_PATH)
        llm = load_llm()
        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

        def conversational_chat(query):
            result = chain({"question": query, "chat_history": st.session_state['history']})
            st.session_state['history'].append((query, result["answer"]))
            return result["answer"]

        if 'history' not in st.session_state:
            st.session_state['history'] = []

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello! Ask me anything about " + uploaded_file.name + " 🤗"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey! 👋"]
            
        # Container for the chat history
        response_container = st.container()
        # Container for the user's text input
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="Talk to your data here (:", key='input')
                submit_button = st.form_submit_button(label='Send')
                
            if submit_button and user_input:
                output = conversational_chat(user_input)
                
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
