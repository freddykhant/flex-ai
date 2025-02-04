import operator
import json
from typing_extensions import TypedDict, Annotated, List
from dataclasses import dataclass, field
from rag import llm, llm_json_mode, retriever
from prompts import fitness_prompt, router_instructions
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph

# Intialize the graph state
@dataclass(kw_only=True)
class ChatState(TypedDict):
  question : str = field(default=None)
  generation: str = field(default=None)
  query: str = field(default=None)
  documents: List[str] = field(default_factory=list)

# Helper method for formatting documents
def format_docs(docs):
  return "\n\n".join([doc.page_content for doc in docs])

def route_question(state):
  print("\nROUTE QUESTION\n")
  response = llm_json_mode.invoke(
      [SystemMessage(content=router_instructions)]
      + [HumanMessage(content=state["question"])]
  )

  try:
      response_json = json.loads(response.content)
      source = response_json.get("datasource", "generalinfo")  # Default to 'generalinfo' if key is missing
  except json.JSONDecodeError:
      print("\nERROR: Failed to parse JSON response. Defaulting to 'generalinfo'.\n")
      return "generalinfo"

  if source == "generalinfo":
      print("\nROUTING TO GENERAL INFO\n")
      return "generalinfo"
  elif source == "vectorstore":
      print("\nROUTE QUESTION TO RAG\n")
      return "vectorstore"


# Document retrievel
def retrieve(state):
  question = state["question"]
  documents = retriever.invoke(question)
  return{"documents": documents}  

# Generation
def generate(state):
    question = state["question"]
    documents = state.get("documents", [])

    if not documents:
        print("\n[WARNING] No relevant documents found. Possible causes:")
        print("  - Embeddings may not be capturing key terms.")
        print("  - Similarity threshold (0.05) might be too strict.")
        print("  - Text splitting may have removed important context.")
        print("\n👉 Using general LLM response instead...\n")
        
        return {"generation": llm.invoke([HumanMessage(content=question)]).content}

    docs_txt = format_docs(documents)
    fitness_prompt_formatted = fitness_prompt.format(user_question=question, context=docs_txt)
    generation = llm.invoke([HumanMessage(content=fitness_prompt_formatted)])

    return {"generation": generation.content}


# Build LangGraph Workflow
builder = StateGraph(ChatState)
builder.add_node("route_question", route_question)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)

builder.set_conditional_entry_point(
  route_question,
  {
    "generalinfo": "generate",
    "vectorstore": "retrieve"
  }
)
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

graph = builder.compile()

### test code

# input = ChatState(question = "How does range of motion affect hypertophy?") 

# answer = graph.invoke(input)

# print(answer["generation"])