from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from dotenv import load_dotenv
import asyncio

load_dotenv()


async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["mathserver.py"],
                "transport": "stdio",
            },
            "weather": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            },
        }
    )

    tools = await client.get_tools()
    model = ChatOllama(model="llama3.2:latest")

    agent = create_agent(
        model=model,
        tools=tools,
    )

    # key change: use ainvoke instead of invoke
    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "what's (3+5)*12?",
                }
            ]
        }
    )

    print("Maths response:", result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())

'''
Async (asynchronous) programming is a way of writing code so that a program can start a slow task (like a network call or file read), then do other work while it waits, instead of blocking until that task finishes. In Python, this is mainly done with the async and await keywords plus an event loop (for example via asyncio), which lets many waiting tasks be managed concurrently in a single thread.

This is especially useful for I/O-bound work such as API calls, database queries, or handling many web requests, where most of the time is spent waiting for external systems rather than using the CPU. It is different from multithreading or true parallelism: async code is still typically single-threaded, but it interleaves tasks intelligently so that the program does not sit idle during waits.

'''
