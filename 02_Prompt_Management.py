from langchain_core.prompts import PromptTemplate

# simple prompt
prompt = PromptTemplate.from_template("Explain about {topic} in 2 sentences")

formatted_template = prompt.invoke({"topic":"Langchain"})

print(formatted_template.to_string())


formatted_template = prompt.invoke({"topic":"Vector database"})

print(formatted_template.to_string())


#Multiple variable
prompt = PromptTemplate.from_template(
    """
You are a trainer.

Student Name: {name}
Technology: {technology}

Provide a beginner-friendly learning path.
"""
)

print("="*60)

formatted_template = prompt.invoke(
    {
        "name":"Rajesh",
        "technology": "Langchain"
    }
)

print(formatted_template.to_string())

#prompt with llm

