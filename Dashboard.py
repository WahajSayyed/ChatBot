import streamlit as st

st.set_page_config(
    page_title="Dashboard",
    page_icon="👋",
)

st.write("# Welcome to the Intelligent Chatbot Application! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
Welcome to our **Intelligent Chatbot Application**! This project harnesses the power of open-source large language models like **LLaMA** and **Gemma** to provide you with a versatile, AI-driven conversational agent. Whether you need quick answers to general queries or want to dive deep into the content of your PDF documents, our chatbot is here to assist you.

## Key Features:

- **Interactive Chat Interface**: Engage in real-time conversations with the chatbot. Ask questions, seek advice, or simply explore what AI can offer in a natural language setting.

- **PDF Query System**: Upload your PDF documents and let the chatbot retrieve relevant information based on your queries. This feature is perfect for students, researchers, and professionals looking to extract key insights from text-heavy documents.

- **Advanced NLP**: Our application leverages cutting-edge Natural Language Processing (NLP) techniques to understand and respond to your inputs with contextual accuracy.

- **User-Friendly Design**: Built with **Streamlit**, the application offers an intuitive and responsive interface that makes navigating and using the chatbot easy and enjoyable.

## Technologies Behind the Scenes:

- **Ollama Platform**: Manages and deploys the large language models that power the chatbot.
- **Streamlit**: Provides a sleek and interactive user interface.
- **Chroma DB**: Serves as the vector store, enabling efficient text indexing and retrieval from PDF documents.

---

Feel free to explore the features and capabilities of this intelligent chatbot. We hope you find it both useful and engaging!

*Happy chatting!*

"""
)