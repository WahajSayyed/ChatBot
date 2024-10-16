from langchain_community.chat_models import ChatOllama
import streamlit as st
from streamlit_chat import message
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama


#llm = Ollama(model="mistral")
#llm = ChatOllama(model="llama3.1")  #ChatOllama(model="mistral")


with st.sidebar:""

st.title("💬 Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():

    llm = ChatOllama(model="gemma:2b") #OpenAI(api_key=openai_api_key)
    prompt_tmplt = PromptTemplate.from_template(
        """
        <s> [INST]You are an assistant for question-answering tasks. [/INST] </s> 
        [INST] Question: {question} 
        Answer: [/INST]
        """
    )
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Get the response from the model
    #response = llm.invoke(input=prompt)
    chain = ({"question": RunnablePassthrough()}
                      | prompt_tmplt
                      | llm
                      | StrOutputParser())
    
    # Parse the response using StrOutputParser
    #query = prompt
    msg = chain.invoke(input=prompt)
    #msg = StrOutputParser().parse(response)
    
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
