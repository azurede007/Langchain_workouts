from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


prompt = PromptTemplate.from_template(
    """
You are a trainer.

Student Name: {name}
Technology: {technology}

Provide a beginner-friendly learning path.
"""
)

seqsteps = prompt | llm

inputvar =  {"name":"Rajesh", "technology": "Langchain" }

response = seqsteps.invoke(inputvar)

print(response.content)

