from langchain.agents import create_agent
from langchain_ollama import ChatOllama

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

def main():
    # Usamos qwen3:8b que soporta herramientas (tools)
    llm = ChatOllama(model="qwen3:8b", temperature=0)
    
    agent = create_agent(
        model=llm,
        tools=[get_weather],
        system_prompt="You are a helpful assistant",
    )

    # Run the agent
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
    )
    print(result)

if __name__ == "__main__":
    main()
