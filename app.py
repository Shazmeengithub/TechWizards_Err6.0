from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize the FastAPI app
app = FastAPI(
    title="Diagnosys API",
    description="Backend API for the Diagnosys AI Medical Consultation Assistant.",
    version="1.0.0"
)

# Load the fine-tuned model and tokenizer
def load_fine_tuned_model():
    """
    Load the fine-tuned model and tokenizer from the local directory.
    """
    print("Loading fine-tuned model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained("fine_tuned_model")
    tokenizer = AutoTokenizer.from_pretrained("fine_tuned_model")
    print("Model and tokenizer loaded successfully!")
    return model, tokenizer

model, tokenizer = load_fine_tuned_model()

# Define a Pydantic model for user input validation
class UserQuery(BaseModel):
    message: str  # User's input message

# Chat endpoint to handle user queries
@app.post("/chat")
async def handle_chat(query: UserQuery = Body(...)):
    """
    Processes user messages and returns the chatbot's response.
    """
    try:
        # Tokenize the user's input
        inputs = tokenizer(query.message, return_tensors="pt")

        # Generate the chatbot's response
        outputs = model.generate(
            inputs.input_ids,
            max_length=512,  # Adjust as needed
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode the response
        bot_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {"response": bot_response}
    except Exception as e:
        # Handle errors gracefully
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your request: {str(e)}"
        )

# Run the app with: uvicorn app:app --reload