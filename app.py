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