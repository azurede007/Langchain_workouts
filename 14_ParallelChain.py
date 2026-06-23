from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


from dotenv import load_dotenv

load_dotenv()

# 1. Initialize our LLM (Gemini Flash)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
parser = StrOutputParser()


sentiment_prompt = PromptTemplate.from_template(
    "Analyze the sentiment of the following product review. Respond with exactly one word: Positive, Negative, or Neutral.\n\nReview: {review}"
)

sentiment_chain = sentiment_prompt | llm | parser

sample_review = """
I bought this wireless keyboard last week. The battery life is phenomenal and it feels great to type on, 
but the Bluetooth connection randomly drops every 20 minutes, which makes gaming completely impossible. 
Returning it tomorrow.
"""

#response = sentiment_chain.invoke({"review": sample_review})

#print(response)

summary_prompt = PromptTemplate.from_template(
    "Summarize the following product review in a single sentence emphasizing the main complaint or praise.\n\nReview: {review}"
)

summary_chain = summary_prompt | llm | parser

parallel_chain = RunnableParallel(predict_sentiment = sentiment_chain, review_summary = summary_chain)

response = parallel_chain.invoke({"review": sample_review})

print("Result:")

print("Sentiment:",response["predict_sentiment"])
print("Summary:",response["review_summary"])

