import streamlit as st
import json
from graph import graph, ChatState

# streamlit page config
st.set_page_config(page_title="Flex AI", page_icon="💪", layout="wide")

# title and intro
st.title("💪Flex AI: Your AI Personal Trainer")
st.write("Ask me anything about hypertophy, resistance training, nutrition, and muscle growth!")

# user input
user_question = st.text_input("Enter your question:")

# process user input and get response
if st.button("Get Answer"):
  if user_question:
    # run the RAG pipeline
    input_state = ChatState(question=user_question)
    response = graph.invoke(input_state)  

    # extract response
    answer = response.get("generation", "no response generated.")
    st.subheader("🤖Flex's Answer:")
    st.write(answer)

    # show retrieved documents 
    documents = response.get("documents", [])
    if documents:
      with st.expander("📚 Relevant Research Documents (click to expand)"):
        for i, doc in enumerate(documents):
          st.write(f"**Document {i+1}:** {doc.page_content[:500]}...")  # Show snippet

  else:
    st.warning("⚠️Please enter a question before submitting.")