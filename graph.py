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
  route_question = llm_json_mode.invoke(
    [SystemMessage(content=router_instructions)]
    + [HumanMessage(content=state["question"])]
  )
  source = json.loads(route_question.content)["datasource"]
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
  documents = state["documents"]
  docs_txt = format_docs(documents)

  # Format the prompt with the user's question
  fitness_prompt_formatted = fitness_prompt.format(context = docs_txt, user_question=question)
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

input = ChatState(question = "What is hypertrophy training?") 

answer = graph.invoke(input)

print(answer["generation"])