from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

prompt = PromptTemplate.from_template("Explain {topic} in 2 simple terms.")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.7)

parser = StrOutputParser()

chain = prompt | llm | parser 

response = chain.invoke({"topic": "mysql database"})

print(response)