from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.embeddings import OllamaEmbeddings
import os

class ChatPDF:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.vector_store = None
        self.retriever = None
        self.chain = None
        self.model = None
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST]You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.  [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )
        self.initialize_model()
        self.load_vector_store()


    def initialize_model(self):
        self.model = ChatOllama(model="gemma:2b")

    def load_vector_store(self):
        if os.path.exists(self.persist_directory):
            self.vector_store = Chroma(persist_directory=self.persist_directory, embedding_function=OllamaEmbeddings(model="llama3.1"))
            self.update_retriever()

    def ingest(self, file_path: str):
        try:
            if file_path.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path=file_path)
                docs = loader.load()
            elif file_path.lower().endswith('.txt'):
                # Open text file with proper encoding
                loader = TextLoader(file_path)
                docs = loader.load()
                # with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                #     content = f.read()
                #     docs = [{"text": content, "metadata": {"source": file_path}}]
            else:
                raise ValueError(f"Unsupported file format: {file_path}. Only PDF and TXT files are supported.")

            # if file_path.lower().endswith('.txt'):
            #     chunks = self.text_splitter.split_text(docs[0]["text"])
            #     chunks = [{"text": chunk, "metadata": {"source": file_path}} for chunk in chunks]
            # else:
            chunks = self.text_splitter.split_documents(docs)
            chunks = filter_complex_metadata(chunks)

            # Ingest the chunks into the vector store
            if self.vector_store is None:
                self.vector_store = Chroma.from_documents(
                    documents=chunks, 
                    embedding=OllamaEmbeddings(model="llama3.1"),
                    persist_directory=self.persist_directory
                )
            else:
                self.vector_store.add_documents(chunks)

            # Persist the vector store
            self.vector_store.persist()
            self.update_retriever()

        except Exception as e:
            raise Exception(f"Error ingesting file {file_path}: {str(e)}")


    def update_retriever(self):
        if self.vector_store:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5},
            )
            self.chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.model
                | StrOutputParser()
            )

    def remove_document(self, file_path: str):
        if self.vector_store:
            self.vector_store.delete(where={"source": file_path})
            self.vector_store.persist()
            self.update_retriever()

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a document first."
        return self.chain.invoke(query)

    def clear(self):
        if self.vector_store:
            self.vector_store.delete_collection()
            self.vector_store = None
        self.retriever = None
        self.chain = None
