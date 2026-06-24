import warnings

# 1. Suppress the deprecation warnings completely
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Import the entity memory class
from langchain_classic.memory import ConversationEntityMemory
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

# 2. Initialize Gemini Flash
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# 3. Define the specialized template
# This memory type requires standard string inputs (not MessagesPlaceholder)
# because it passes a dictionary of entities alongside the history.
prompt_template = ChatPromptTemplate.from_template("""
You are a friendly chatbot. Keep answers short.

Context about entities mentioned so far:
{entities}

Recent conversation history:
{chat_history}

User: {input}
AI:""")

# 4. Initialize Entity Memory
# Like summary memory, entity memory needs an LLM to extract entities in the background.
memory = ConversationEntityMemory(
    llm=llm,
    chat_history_key="chat_history",
    story_key="entities"
)

parser = StrOutputParser()


# 5. Build the LCEL Chain
# We use a custom function to load both variables from the memory object dynamically
def get_memory_variables(input_dict):
    user_input = input_dict["input"]
    # load_memory_variables requires the current input so it knows which entities to look up
    return memory.load_memory_variables({"input": user_input})


chain = (
        {
            "input": RunnablePassthrough(),
            # Fetch 'chat_history' and 'entities' from memory simultaneously
            "chat_history": lambda x: get_memory_variables({"input": x})["chat_history"],
            "entities": lambda x: get_memory_variables({"input": x})["entities"]
        }
        | prompt_template
        | llm
        | parser
)

# 6. Chat Loop
print("Chatbot Initialized! (Extracting and tracking specific entities)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Run the chain
    response = chain.invoke(user_input)
    print(f"AI: {response}\n")

    # Save the current turn to memory.
    # This triggers an internal LLM call to extract any entities from the user's input.
    memory.save_context({"input": user_input}, {"output": response})

    # Optional: Debug line to view the extracted entity store
    # print(f"--- DEBUG ENTITY STORE: {memory.entity_store} ---\n")