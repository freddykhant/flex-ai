import operator
from typing_extensions import TypedDict, Annotated, List
from dataclasses import dataclass, field
from rag import llm, retriever
from prompts import fitness_prompt
from langchain_core.messages import HumanMessage
from langgraph.graph import START, END, StateGraph

# Intialize the graph state
@dataclass(kw_only=True)
class ChatState(TypedDict):
  topic : str = field(default=None)
  generation: str = field(default=None)
  query: str = field(default=None)
  doucments: List[str] = field(default_factory=list)

# Helper method for formatting documents
def format_docs(docs):
  return "\n\n".join([doc.page_content for doc in docs])

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

  fitness_prompt_formatted = fitness_prompt.format(context=docs_txt, topic=question)
  generation = llm.invoke([HumanMessage(content=fitness_prompt_formatted)])

  return {"generation": generation.content}

# Build LangGraph Workflow
builder = StateGraph(ChatState)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

graph = builder.compile()

