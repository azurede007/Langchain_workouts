from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


from dotenv import load_dotenv

load_dotenv()

# 1. Initialize our LLM (Gemini Flash)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
parser = StrOutputParser()

# ==========================================
# CHAIN 1: The Naming Chain
# ==========================================
prompt_one = PromptTemplate.from_template(
    """
    You are a branding expert. Generate one single, creative, catchy name for a business that does this: {business_description}. 
    Only return the name, nothing else."""
)
# This sub-chain outputs a clean string representing the company name
chain_one = prompt_one | llm | parser

business_idea = "Building AI app for Travel guide. Get location as input from the user and it help to get the info about the place"

#response = chain_one.invoke({"business_description":business_idea})

#print(response)


# ==========================================
# CHAIN 2: The Slogan Chain
# ==========================================
prompt_two = PromptTemplate.from_template(
    "You are a copywriter. Write a one-sentence punchy marketing slogan for a company named: {company_name}."
)

# This sub-chain takes a company name and outputs a slogan string
chain_two = prompt_two | llm | parser

multi_chain = ({"company_name": chain_one} | chain_two)

final_slogan = multi_chain.invoke({"business_description":business_idea})

print(final_slogan)
