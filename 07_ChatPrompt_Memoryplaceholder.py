from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ]
)

messages = prompt.invoke(
    {
        "chat_history": [
            ("human", "What is Databricks?"),
            ("ai", "Databricks is a data processing framework")
        ],
        "question": "Brief about partitioning"
    }
)

print(messages)