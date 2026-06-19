from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a technical trainer."),
        ("human",
         """
         Student Name: {name}
         Technology: {technology}

         Provide a beginner-friendly learning path.
         """
        )

    ]
)

chain = prompt | llm


inputvar =  {"name":"Rajesh", "technology": "Langchain" }

response = chain.invoke(inputvar)

print(response.content)
