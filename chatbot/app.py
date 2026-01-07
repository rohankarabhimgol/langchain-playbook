from langchain_openai import ChatOpenAI
# from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# LangSmit Tracking
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

## Prompt Template

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpfull assistant. Please respond to the user queries"),
        ("user","Question:{question}")
    ]
)

## Streamlit freamework

st.title("LangChain Demo with OPENAI API")
input_text = st.text_input("Search the topic you want ")

# OpenAI LLM
llm = ChatOpenAI(model = "gpt-3.5-turbo")
# llm = OllamaLLM(model = "llama3.2")
output_parser = StrOutputParser()
chain = prompt_template|llm|output_parser

if input_text:
    response = chain.invoke({"question":input_text})
    st.write(response)

# streamlit run app.py