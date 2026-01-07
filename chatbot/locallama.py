from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv
import time

load_dotenv()

# Optional LangSmith tracking
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Check available models
with st.sidebar:
    st.header("Configuration")
    # Create dropdown with hardcoded options
    model_name = st.selectbox(
        "Choose a model:",
        options=["llama3.2", "llama3:latest", "deepseek-coder:latest", "deepseek-coder:1.3b", "phi3", "mistral"],
        index=0  # Default selection
    )
    st.caption("Check with: `ollama list`")

st.title("🤖 LangChain Demo with Local LLM")
st.markdown(f"Using Ollama with {model_name}")

# Input from user
input_text = st.text_area("Ask your question:", height=100)

if st.button("Get Answer") and input_text:
    with st.spinner(f"Thinking with {model_name}..."):
        try:
            # Create components
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant."),
                ("user", "{question}")
            ])
            
            llm = OllamaLLM(
                model=model_name,
                temperature=0.7,
                timeout=60  # 60 second timeout
            )
            
            # Create and invoke chain
            chain = prompt | llm | StrOutputParser()
            
            start_time = time.time()
            response = chain.invoke({"question": input_text})
            end_time = time.time()
            
            # Display results
            st.success(f"✅ Response generated in {end_time-start_time:.1f} seconds")
            
            with st.expander("See full response", expanded=True):
                st.write(response)
                
        except Exception as e:
            st.error(f"🚨 Error: {str(e)}")
            st.markdown("""
            **Troubleshooting steps:**
            1. Open a terminal and run: `ollama serve`
            2. Make sure the model is downloaded: `ollama pull llama3.2`
            3. Check if Ollama is responding: `ollama list`
            """)