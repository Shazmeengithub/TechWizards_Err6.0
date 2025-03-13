import streamlit as st
import google.generativeai as genai
import pandas as pd
import re

# Set API Key
genai.configure(api_key="")

# Initialize model
model = genai.GenerativeModel("gemini-pro")

# Custom CSS for styling
st.markdown(
    """
    <style>
    [class="st-emotion-cache-128upt6 eht7o1d3"] {
           background: linear-gradient(to right, #232526, #414345);
            color: white;
        }
        header {visibility: hidden;}
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(to right, #232526, #414345);
            color: white;
        }
        .title-container {
            text-align: center;
            padding: 15px;
            font-size: 30px;

            color: white;
            background: #6D6975;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .chat-container {
            display: flex;
            align-items: center;
            padding: 10px 0;
        }
        .stChatMessage {
            border-radius: 12px;
            padding: 12px;
            margin: 5px 0;
            font-size: 15px;
            max-width: 70%;
            color: white;
        }
        .user {
            background-color: #36454F;
            margin-left: auto;
            text-align: right;
        }
        .assistant {
            background-color: #6D6975;
            margin-right: auto;
            text-align: left;
        }
        .chatbox {
            padding: 20px;
            background: #232526;
            border-radius: 12px;
            margin-bottom: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown('<div class="title-container">Find Your Doctor 🏥</div>', unsafe_allow_html=True)

# Chat Input
user_input = st.chat_input("Ask to find Doctors near you...")

def parse_doctor_info(text):
    """Parses structured doctor information into a list of dictionaries with consistent formatting."""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # Remove bold formatting from names
    text = re.sub(r"\s*-\s*", "\n", text)  # Ensure multi-line format for details

    doctor_entries = re.split(r"\nDr\.\s", text)
    doctor_data = []

    for doctor in doctor_entries:
        if not doctor.strip():
            continue

        lines = doctor.strip().split("\n")
        name = "Dr. " + lines[0].strip()
        details = {"Name": name}

        for detail in lines[1:]:
            if ":" in detail:
                key, value = detail.split(":", 1)
                details[key.strip()] = value.strip()

        doctor_data.append(details)

    return doctor_data

if user_input:
    formatted_query = (
        f"Find nearby '{doctor_type}'doctors from location'{location}'. "
        "Provide structured results with the doctor's name as a heading, followed by Specializations, Degree, Year of Experience, Location, Address, and Contact Info  as bullet points. "
        "Ensure consistency in formatting for all doctors."
    )

    response = model.generate_content(formatted_query)

    # Display user message
    st.markdown(
        f"""
        <div class="chat-container" style="justify-content: flex-end;">
            <div class="stChatMessage user">{user_input}</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Check if AI response contains a markdown table or bullet points
    doctor_data = parse_doctor_info(response.text)
    
    for doctor in doctor_data:
        st.markdown(f"### {doctor['Name']}")  # Ensure name is always a heading
        for key, value in doctor.items():
            if key != "Name":
                st.markdown(f"- **{key}:** {value}")

