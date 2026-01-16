import os
import torch
import json
import Evaluate
import Agent_MS
from transformers import BertTokenizer

# Ensure we are in the correct directory (or handle paths relatively)
# This script assumes it's placed in one of the domain directories (e.g., Literal Creation)
# or you need to adjust paths. For simplicity, let's assume it runs in "Literal Creation".

def chat_interactive(pool_file="LiteralPool.jsonl", model_path="model.pth", optimizer_path="optimizer.pth"):
    print("Loading memory and model...")
    
    # Check if files exist
    if not os.path.exists(pool_file):
        print(f"Error: Memory pool '{pool_file}' not found.")
        return
    if not os.path.exists(model_path):
        print(f"Error: Model '{model_path}' not found. Please run Integrate.py first.")
        return

    # Load all memories
    all_sentences = Evaluate.allSentences(pool_file)
    print(f"Loaded {len(all_sentences)} memories.")
    print("System ready! Type 'exit' to quit.\n")

    while True:
        user_input = input("\n[User]: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        if not user_input.strip():
            continue

        print("Thinking (Retrieving memories & Generating answer)...")
        
        # 1. Retrieval: Find best memories using the trained BERT model
        # final_prompt combines the user question with the best retrieved memories
        try:
            final_question = Evaluate.final_prompt(user_input, model_path, optimizer_path, all_sentences)
            
            # Show what the LLM actually sees (Transparency)
            # print(f"\n[Prompt Sent to LLM]:\n{final_question}\n")
            
            # 2. Generation: Call LLM (DeepSeek via SiliconFlow)
            answer = Evaluate.chatgpt_answer(final_question)
            
            print(f"[Agent]: {answer}")
            
            # Optional: Save good interactions? 
            # In interactive mode, we might not want to auto-save unless confirmed.
            # But the original logic does:
            # Agent_MS.message_store(final_question, answer, pool_file, "Literature")
            
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    # You can change these paths to point to Logic/Plan/OnePool folders if needed
    # Defaulting to Literal Creation context
    chat_interactive()
