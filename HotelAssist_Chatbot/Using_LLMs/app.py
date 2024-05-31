from flask import Flask, render_template, request, jsonify
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class GPT2Chatbot:
    def __init__(self, model_path, tokenizer_path):
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def generate_response(self, input_text):
        with torch.no_grad():
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            output = self.model.generate(input_ids, max_length=150, num_beams=5, temperature=0.7, top_k=50, top_p=0.95)
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return response

BASE_DIR = "/Users/vishalpatel/Desktop/Data Science/Chatbot/data_folder/test18/HotelAssist/immigration/HotelAssist/Bot"
chatbot = GPT2Chatbot(BASE_DIR, BASE_DIR)

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/chat', methods=['POST'])
# def chat():
#     try:
#         user_message = request.form.get('user_message')
#         if not user_message:
#             raise ValueError("User message is empty or not provided.")
#         print(f"User Input: {user_message}")  # Printing user input
#         bot_response = chatbot.generate_response(user_message)
#         return jsonify({'bot_response': bot_response})
#     except Exception as e:
#         print(f"Error occurred: {e}")
#         return jsonify({'bot_response': f'Sorry, I encountered an error: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.form.get('user_message')
        if not user_message:
            raise ValueError("User message is empty or not provided.")
        print(f"User Input: {user_message}")  # Printing user input
        bot_response = chatbot.generate_response(user_message)
        return jsonify({'bot_response': bot_response})
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'bot_response': f'Sorry, I encountered an error: {str(e)}'}), 500


@app.errorhandler(400)
def bad_request_error(e):
    return jsonify({'error': 'Bad Request', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
