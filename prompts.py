from langchain.prompts import PromptTemplate

fitness_prompt = PromptTemplate.from_template(
    "You are a fitness expert specializing in hypertrophy training. A user asks: '{user_question}'. Provide a detailed and informative response."
)
