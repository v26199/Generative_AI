# LLMs_finetuned_chatapp
ChatApplication using Llama 2, Sentence Transformers, CTransformers, Langchain, and Streamlit.

## Overview
This project is a chat application for interacting with custom data using conversational AI. It allows users to upload data in various formats such as CSV, JSON, TXT, or PDF, and then engage in conversation with the uploaded data.

## Dependencies
- `streamlit`: For building the web application interface.
- `streamlit_chat`: For displaying chat messages in the interface.
- `PyMuPDF`: For handling PDF files.
- `langchain_community`: For document loading, embeddings, vector stores, and language models.
- `transformers`: For loading and utilizing language models.

## Usage
1. Install dependencies using `pip install -r requirements.txt`.
2. Run the application using `streamlit run app.py`.
3. Upload your data in CSV, JSON, TXT, or PDF format.
4. Start conversing with your data by typing queries into the input field and hitting send.

## File Handling
- `load_file(file)`: Function to load different file types (CSV, JSON, TXT, PDF) and return their content.
- Uploaded files are processed and converted into a format suitable for conversation.

## Model Loading
- `load_llm()`: Function to load the conversational language model.
- The model used is a custom conversational retrieval model trained on the specified data.

## Conversation
- Users can input queries into the text input field and send them.
- The system responds with relevant answers based on the uploaded data and conversation history.
- Conversation history is displayed in a chat-style interface.

## Persistence
- User's chat history and generated responses are stored in the session state for continuity during the session.
- Past interactions and responses are displayed in the interface.

## Error Handling
- Unsupported file types are handled with an error message.
