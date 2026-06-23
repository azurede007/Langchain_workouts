from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from dotenv import load_dotenv

load_dotenv()

# 1. Initialize Gemini Flash and Gemini Embeddings
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# 2. Seed some sample data into an In-Memory Chroma DB
# (For a permanent DB, add the parameter: persist_directory="./my_chroma_db")
sample_docs = [
    Document(page_content="The company password change policy requires updates every 90 days using 12 characters."),
    Document(page_content="To apply for remote work, employees must fill out Form 404-B on the HR portal."),
    Document(page_content="The office kitchen fridge is cleaned out every Friday at 4:00 PM; unlabelled items are tossed."),
    Document(page_content="Casual Leave=5 days,Floating Holidays=2 days, Sick Leave=5 days")

]

vector_store = Chroma.from_documents(
    documents=sample_docs,
    embedding=embeddings,
    collection_name="company_policy"
)

# 3. Turn the vector store into a Retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 4. Construct the RAG Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an internal HR assistant. 
                  Answer the user's question using ONLY the provided context below. 
                  If you don't know, say you don't know.
                  
                  Context:{context}
                  
                  """),
    ("human", "{question}")
])

# 5. Build the LCEL RAG Chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("Give info about leave policy")

print(response)