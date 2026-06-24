import warnings

# 1. Suppress the deprecation warnings completely
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import the modern Chroma integration and legacy Vector memory
from langchain_chroma import Chroma
from langchain_classic.memory import VectorStoreRetrieverMemory
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

# 2. Initialize Gemini Flash and Gemini Embeddings
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# 3. Initialize ChromaDB Vector Store
# By default, leaving out `persist_directory` runs Chroma transiently in RAM.
# To save history permanently to your computer, uncomment the line below:
vectorstore = Chroma(
    collection_name="chat_history_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db_mail_storage"
)

# 4. Set up the Retriever Memory
# k=2 tells Chroma to retrieve the 2 most semantically matching past exchanges
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
memory = VectorStoreRetrieverMemory(retriever=retriever, memory_key="history")

# 5. Define prompt template expecting a text 'history' block
prompt_template = ChatPromptTemplate.from_template("""
You are a friendly chatbot. Keep answers short.

Relevant context from previous conversations:
{history}

User: {input}
AI:""")

parser = StrOutputParser()

# 6. Build the LCEL Chain
chain = (
        {
            "input": RunnablePassthrough(),
            # Chroma searches context records using the string similarity of the current input
            "history": lambda x: memory.load_memory_variables({"prompt": x})["history"]
        }
        | prompt_template
        | llm
        | parser
)

# 7. Chat Loop
print("Chatbot Initialized! (Vector Memory enabled via ChromaDB)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Run the chain
    response = chain.invoke(user_input)
    print(f"AI: {response}\n")

    # Save context (Text converts to an embedding and saves immediately to Chroma)
    memory.save_context({"input": user_input}, {"output": response})