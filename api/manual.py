"""
Simpler Version - With Manual Route to Understand Better
"""

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langserve import add_routes
import uvicorn

app = FastAPI(title="Super Simple API")

# 1. Setup model
llm = OllamaLLM(model="llama3.2")

# 2. Create one template
greet_prompt = ChatPromptTemplate.from_template(
    "Say hello to {name} in a {style} way"
)

# 3. Auto-route (LangServe magic) - REQUIRES {"input": {...}} format
add_routes(
    app,
    greet_prompt | llm,
    path="/auto-greet"
)

# 4. Manual route (You control everything)
@app.post("/manual-greet")
async def manual_greet(name: str, style: str = "friendly"):
    """
    Manual endpoint - shows you what happens inside
    """
    # Create the prompt manually
    prompt_text = f"Say hello to {name} in a {style} way"
    
    # Call the model
    response = llm.invoke(prompt_text)
    
    return {
        "endpoint": "manual",
        "prompt_created": prompt_text,
        "ai_response": response
    }

# 5. Simple GET endpoint
@app.get("/")
async def home():
    return {
        "message": "Simple API is working!",
        "endpoints": {
            "auto_route": "POST /auto-greet/invoke",
            "manual_route": "POST /manual-greet?name=John&style=friendly",
            "docs": "GET /docs"
        }
    }

if __name__ == "__main__":
    print("\nSimple API with 2 ways to do the same thing:")
    print("1. AUTO: /auto-greet - LangServe does the work")
    print("2. MANUAL: /manual-greet - You see how it works")
    print("\nOpen: http://localhost:8000/docs\n")
    
    uvicorn.run(app, host="localhost", port=8000)




# http://localhost:8000/auto-greet/invoke
# {
#   "input": {
#     "name": "Alice",
#     "style": "funny"
#   }
# }

# http://localhost:8000/manual-greet?name=Bob&style=formal