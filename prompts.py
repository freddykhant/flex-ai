from langchain.prompts import PromptTemplate

router_instructions = """You are an expert at routing a user question to a vectorstore or general query.

The vectorstore contains spreadsheets related to the fitness, hypertrophy, rest, and nutrition.

Use the vectorstore for questions on these topics. For all else, use trained/general information.

Return JSON with ONLY single key, datasource, that is 'generalinfo' or 'vectorstore' depending on the question."""

fitness_prompt = PromptTemplate.from_template(
    "You are a fitness expert specializing in hypertrophy training. A user asks: '{user_question}'. Provide a detailed and informative response."
)
