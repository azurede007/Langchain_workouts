from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

@tool
def multiply(a:int,b:int)->int:
    """Multiply two numbers"""
    return (a * b) + 10

@tool
def add(a:int,b:int)->int:
    """Add two numbers"""
    return a + b

@tool
def sub(a:int,b:int)->int:
    """Subtract two numbers"""
    return a + b

@tool
def get_ticket_count(status: str) -> str:
    """Get ticket count by status"""
    data = {
        "open": 125,
        "closed": 500
    }
    return str(data.get(status.lower(), 0))

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

prompt = ChatPromptTemplate.from_template("""
You are a friendly chatbot. Keep answers short.
User: {input}
Placeholder: {agent_scratchpad}
AI:""")

tools = [add, sub, multiply,get_ticket_count]

# Agent
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False
)
parser = StrOutputParser()

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Run the chain
    response = agent_executor.invoke({"input": user_input})
    final_output = parser.parse(response["output"][0]["text"])
    print(f"AI: {final_output}\n")



