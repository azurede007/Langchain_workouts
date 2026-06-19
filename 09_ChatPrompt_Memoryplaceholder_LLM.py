from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv()

# 1. Initialize Gemini Flash
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

# 2. Define the template with the history placeholder
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Keep your answers concise."),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{current_question}")
])

# 3. Create the chain
chain = prompt_template | llm

# 4. Initialize an empty list to store the ongoing chat history
chat_history = []

print(" Chatbot initialized! Type 'exit' or 'quit' to end the conversation.\n")

# 5. Start the interaction loop
while True:
    # Get input from the user via terminal
    user_input = input("You: ")

    # Check for exit condition
    if user_input.strip().lower() in ['exit', 'quit']:
        print("Goodbye!")
        break

    # Skip empty inputs
    if not user_input.strip():
        continue

    # Invoke the chain, passing the history list and the new question
    response = chain.invoke({
        "history": chat_history,
        "current_question": user_input
    })

    # Print the model's response
    print(f"AI: {response.content}\n")

    # CRITICAL STEP: Append both the question and the answer to history
    # so the model remembers them in the next loop iteration
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response.content))