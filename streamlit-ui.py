import streamlit as st
from RAG.rag import RAG_chatbot

# Initialize chatbot
bot = RAG_chatbot()

# Streamlit UI setup
st.set_page_config(page_title="Medical Chatbot", layout="centered")
st.title("Chat with Medical Assistant")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    role, text = message
    with st.chat_message(role):
        st.write(text)

# User input
user_query = st.text_input("Type your message here...")
if st.button("Send") and user_query:
    # Display user message
    st.session_state.messages.append(("user", user_query))
    with st.chat_message("user"):
        st.write(user_query)

    # Get chatbot response
    response = bot.get_response(user_query)
    st.session_state.messages.append(("assistant", response))
    with st.chat_message("assistant"):
        st.write(response)
