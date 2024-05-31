# Testing model performance

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GPT2Chatbot:
    def __init__(self, model_path, tokenizer_path):
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def generate_response(self, input_text):
        with torch.no_grad():  # ensures no gradient computation for faster inference
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            output = self.model.generate(input_ids, max_length=150, num_beams=5, temperature=0.7, top_k=50, top_p=0.95)
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return response

if __name__ == "__main__":
    BASE_DIR = "/Users/vishalpatel/Desktop/Data Science/Chatbot/data_folder/test18/HotelAssist/immigration/HotelAssist/Bot"  # adjust as per your directory structure
    chatbot = GPT2Chatbot(BASE_DIR, BASE_DIR)

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting the chat. Goodbye!")
            break
        response = chatbot.generate_response(user_input)
        print(f"Bot: {response}")

