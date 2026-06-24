
import warnings

from langchain_classic.memory import ConversationBufferMemory, ConversationBufferWindowMemory

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
# NEW: Import the modern history utilities


from dotenv import load_dotenv

load_dotenv()

# 1. Initialize Gemini Flash
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# 2. Define a template with a specific MessagesPlaceholder for history
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly chatbot. Keep answers short."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

parser = StrOutputParser()

memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True,k=2)

# 3. Wrap your chain with message history management
chain = (
        {
            "input": RunnablePassthrough(),
            "chat_history": lambda x: memory.load_memory_variables({})["chat_history"]
        }
        | prompt_template
        | llm
        | parser
)

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    response = chain.invoke(
        {"input": user_input}
    )
    print(response)
    memory.save_context({"input": user_input}, {"output": response})