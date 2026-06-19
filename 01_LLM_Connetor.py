"""

pip install langchain-google-genai
pip install langchain-openai
pip install langchain-anthropic
pip install langchain-ollama
pip install langchain-mistralai

"""

from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai


from dotenv import load_dotenv

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

response = llm.invoke("Explain about Langchain in 2 sentances")

print("="*60)
print(response.content)
print("="*60)

# Initialize client with API key
client = genai.Client(api_key="YOUR_GOOGLE_API_KEY")

# Generate response
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain about LangChain in 2 sentences"
)

print(response.text)


#Open AI Models
from langchain_openai import ChatOpenAI
#llm = ChatOpenAI(model="gpt-4o",api_key ="API_KEY")
llm = ChatOpenAI(model="gpt-4o")

response = llm.invoke("Explain about Langchain in 2 sentances")

print("="*60)
print(response.content)
print("="*60)
