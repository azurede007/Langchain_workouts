from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant."),
        ("human", "Explain {topic} in simple terms.")
    ]
)


formatted_prompt = prompt.invoke(
    {"topic": "Vector Databases"}
)

print(formatted_prompt)

#Multiple variable
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

formatted_template = prompt.invoke(
    {
        "name":"Rajesh",
        "technology": "Langchain"
    }
)

print(formatted_template.to_string())