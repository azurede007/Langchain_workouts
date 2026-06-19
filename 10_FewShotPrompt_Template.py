
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate
)
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# 1. Initialize Gemini Flash
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.1)

# 2. Provide few-shot examples for the model to learn from
examples = [
    {
        "ticket": "Production database is down",
        "priority": "Critical"
    },
    {
        "ticket": "Unable to login to portal",
        "priority": "Medium"
    },
    {
        "ticket": "Need password reset",
        "priority": "Low"
    }
]

# 3. Define the structure for how those examples should look
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "Ticket: {ticket}"),
        ("ai", "Priority: {priority}")
    ]
)

# 4. Assemble the few-shot template wrapper
few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt
)

# 5. Build the final prompt template containing system rules, few-shot examples, and user input
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "Classify ticket priority based on the given examples. Only output the priority level (Low, Medium, High, Critical)."),
        few_shot_prompt,
        ("human", "Ticket: {ticket}")
    ]
)

# 6. Create the chain using LCEL
chain = final_prompt | llm

print("Ticket Priority Classifier Ready! Type 'exit' or 'quit' to stop.\n")

# 7. Start the continuous user input loop
while True:
    user_ticket = input("Enter a support ticket description: ")

    # Check for loop termination
    if user_ticket.strip().lower() in ['exit', 'quit']:
        print("Goodbye!")
        break

    if not user_ticket.strip():
        continue

    # Invoke the chain using the dynamic key '{ticket}'
    response = chain.invoke({
        "ticket": user_ticket
    })

    print(f"Predicted -> {response.content.strip()}\n")