import os
import streamlit as st
from streamlit_chat import message
from RAG1 import ChatPDF

st.set_page_config(page_title="ChatPDF")

# File paths for persistent storage
UPLOAD_FOLDER = "uploaded_files"
CHROMA_PERSIST_DIR = "./chroma_db"

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "assistant" not in st.session_state:
    st.session_state["assistant"] = ChatPDF(persist_directory=CHROMA_PERSIST_DIR)

if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith(('.pdf', '.txt'))]

def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()

def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text)
        
        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))

def read_and_save_file():
    for file in st.session_state["file_uploader"]:
        if file.name not in st.session_state["uploaded_files"]:
            # Save the file
            file_path = os.path.join(UPLOAD_FOLDER, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getvalue())
            
            # Ingest the file
            try:
                with st.spinner(f"Ingesting {file.name}"):
                    st.session_state["assistant"].ingest(file_path)
                st.session_state["uploaded_files"].append(file.name)
                st.success(f"Successfully ingested {file.name}")
            except Exception as e:
                st.error(f"Error ingesting {file.name}: {str(e)}")
                # Remove the file if ingestion failed
                os.remove(file_path)

def delete_file(file_name):
    # Remove file from filesystem
    file_path = os.path.join(UPLOAD_FOLDER, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # Remove file from vector store
    st.session_state["assistant"].remove_document(file_path)
    
    # Remove file from uploaded_files list
    st.session_state["uploaded_files"].remove(file_name)

def page():
    st.header("ChatPDF")

    # Sidebar for file upload and selection
    with st.sidebar:
        st.subheader("Document Management")
        uploaded_file = st.file_uploader(
            "Upload a new document",
            type=["pdf", "txt"],
            key="file_uploader",
            on_change=read_and_save_file,
            label_visibility="collapsed",
            accept_multiple_files=True,
        )

        st.subheader("Available Documents")
        for file in st.session_state["uploaded_files"]:
            col1, col2 = st.columns([3, 1])
            col1.write(file)
            if col2.button("Delete", key=f"delete_{file}"):
                delete_file(file)
                st.experimental_rerun()

    # Main chat interface
    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)

if __name__ == "__main__":
    page()
