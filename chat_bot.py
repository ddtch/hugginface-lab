import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import warnings
warnings.filterwarnings("ignore")

# Optimized for M2 Mac with 16GB RAM
model_name = "google/gemma-3-1b-it"
#"microsoft/DialoGPT-small"  # Better for conversations
device = "mps" if torch.backends.mps.is_available() else "cpu"

pipe = pipeline("text-generation", model=model_name)
messages = [
    {"role": "user", "content": "Who are you?"},
]
response = pipe(messages)
print(response)


# # Initialize the chatbot pipeline
# chatbot = pipeline("text-generation", model=model_name, device=0 if device == "mps" else -1)

# # Store conversation history
# conversation_history = ""

# while True:
#     # Get user input
#     user_input = input("\nYou: ").strip()
    
#     # Exit condition
#     if user_input.lower() in ['quit', 'exit', 'bye']:
#         print("👋 Goodbye!")
#         break
    
#     if not user_input:
#         continue
    
#     # Build conversation context
#     if conversation_history:
#         prompt = f"{conversation_history}\nHuman: {user_input}\nBot:"
#     else:
#         prompt = f"Human: {user_input}\nBot:"
    
#     # Generate response
#     try:
#         response = chatbot(prompt)
        
#         # Extract bot response
#         print(response)
#         full_response = response[0]['generated_text']
#         bot_response = full_response[len(prompt):].strip()
#         print(bot_response)
        
#         # # Clean up response (remove any "Human:" that might appear)
#         # if "Human:" in bot_response:
#         #     bot_response = bot_response.split("Human:")[0].strip()
        
#         # print(f"Bot: {bot_response}")
        
#         # # Update conversation history
#         conversation_history = f"{conversation_history}\nHuman: {user_input}\nBot: {bot_response}"
        
#         # Keep history manageable (last 500 chars)
#         if len(conversation_history) > 500:
#             conversation_history = conversation_history[-500:]
            
#     except Exception as e:
#         print(f"Sorry, I encountered an error: {e}")
#         print("Let's try again!")