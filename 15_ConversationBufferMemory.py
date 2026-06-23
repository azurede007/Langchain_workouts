from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.memory import ConversationBufferMemory

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

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

psrser = StrOutputParser()

chain = (
        {
            "input": RunnablePassthrough(),
            "chat_history": lambda x: memory.load_memory_variables({})["chat_history"]
        }
        | prompt_template
        | llm
        | psrser
)

while True:
    user_input = input("You:")
    response = chain.invoke(user_input)
    print(response)
    memory.save_context({"input": user_input}, {"output": response})
