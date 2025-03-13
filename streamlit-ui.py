import streamlit as st
import requests

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000/chat"  # Update this if your API is hosted elsewhere

# Streamlit UI setup
st.set_page_config(
    page_title="Diagnosys", 
    layout="centered", 
    page_icon="🩺"  # Added a medical-themed icon
)
st.title("AI Medical Consultation Assistant")
st.markdown("""
    Welcome to **Diagnosys**, your AI-powered medical consultation assistant.  
    Ask any health-related questions, and I'll do my best to assist you!
""")  # Added a description for context

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for role, message in st.session_state.chat_history:
    with st.chat_message(role):  # Streamlit's chat message component
        st.write(message)

# User input
user_query = st.chat_input("Type your message here...")  # Streamlit's chat input for better UX

if user_query:
    # Display user message
    st.session_state.chat_history.append(("user", user_query))
    with st.chat_message("user"):
        st.write(user_query)

    # Get chatbot response from the FastAPI backend
    with st.spinner("Thinking..."):  # Show a spinner while processing
        try:
            response = requests.post(API_URL, json={"message": user_query})
            if response.status_code == 200:
                bot_response = response.json().get("response", "No response from the bot.")
                st.session_state.chat_history.append(("assistant", bot_response))
                with st.chat_message("assistant"):
                    st.write(bot_response)
            else:
                error_message = f"Error: {response.json().get('detail', 'Unknown error')}"
                st.session_state.chat_history.append(("assistant", error_message))
                with st.chat_message("assistant"):
                    st.error(error_message)  # Display error in red for visibility
        except Exception as e:
            error_message = f"Failed to connect to the chatbot: {str(e)}"
            st.session_state.chat_history.append(("assistant", error_message))
            with st.chat_message("assistant"):
                st.error(error_message)

# Add a "Clear Chat" button to reset the conversation
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()  # Refresh the app to reflect the cleared chat