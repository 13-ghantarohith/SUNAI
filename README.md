# SUNAI
Creating the answers like open AI
# Install the necessary libraries from Hugging Face for running the model
!pip install transformers accelerate bitsandbytes

print("✅ Libraries installed successfully!")
# This command will prompt you to enter your Hugging Face access token.
# Paste your token and press Enter.
from huggingface_hub import login
login()

print("✅ Logged into Hugging Face successfully!")
import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from getpass import getpass

# 1. Ask for token securely (won’t show when typing/pasting)
HF_TOKEN = getpass("🔑 Enter your Hugging Face token: ")

# 2. Login
login(HF_TOKEN)

# 3. Specify the model
model_id = "google/gemma-2b-it"

# 4. Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN
)

# 5. Create pipeline
chatbot = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)

print("☀️ SUN AI is online and ready to chat!")
# A list to store the history of the conversation
chat_history = []

print("--------------------------------------------------")
print("--- SUN AI ---")
print("Ask me anything! Type 'quit' to exit the chat.")
print("--------------------------------------------------")

while True:
    # Get user input
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        print("SUN AI: Goodbye! Have a great day.")
        break

    # Format the input for the model using its chat template
    # This history helps the model remember the context of the conversation
    messages = chat_history + [{"role": "user", "content": user_input}]
    prompt = chatbot.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Get the AI's response
    outputs = chatbot(
        prompt,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    
    # Clean up and display the response
    response = outputs[0]["generated_text"].split("<start_of_turn>model\n")[-1]
    print(f"SUN AI: {response}")

    # Add the current exchange to history
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": response})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response = outputs[0]["generated_text"].split("<start_of_turn>model\n")[-1]
    output_text = outputs[0]["generated_text"]
# Remove the prompt part, keep only model’s reply
response = output_text[len(prompt):].strip()
messages = [
    {"role": "system", "content": "You are SUN AI, a helpful and friendly assistant, just like ChatGPT."}
]
