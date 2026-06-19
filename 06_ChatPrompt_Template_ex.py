from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
# 1. Initialize the Gemini Flash model
# It automatically looks for the GOOGLE_API_KEY environment variable



model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 2. Define the prompt template with placeholders
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant. Plan a quick itinerary for {destination}."),
    ("user", "I only have {days} days and I love {interest}."),
])

# 3. Chain them together using the pipe (|) operator
chain = prompt_template | model

# 4. Invoke the chain
# The values are injected into the template, and sent to Gemini Flash automatically
response = chain.invoke({
    "destination": "Bangalore",
    "days": "2",
    "interest": "street food and purchase"
})

# 5. Print the result
print(response.content)