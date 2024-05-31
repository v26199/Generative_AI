
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import pickle
from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from keras.preprocessing.text import Tokenizer

app = Flask(__name__)


# ENCODER
class Encoder(nn.Module):
    """Encoder module for the Seq2Seq model."""
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

# DECODER
class Decoder(nn.Module):
    """Decoder module for the Seq2Seq model."""
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim * 2, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

# SEQ2SEQ MODEL
class Seq2Seq(nn.Module):
    """Main Seq2Seq model integrating the Encoder and Decoder."""
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_hidden, encoder_cell = self.encoder(src)

        if self.encoder.rnn.bidirectional:
            hidden = torch.cat((encoder_hidden[0:encoder_hidden.size(0):2],
                                encoder_hidden[1:encoder_hidden.size(0):2]), dim=2)
            cell = torch.cat((encoder_cell[0:encoder_cell.size(0):2],
                              encoder_cell[1:encoder_cell.size(0):2]), dim=2)
        else:
            hidden, cell = encoder_hidden, encoder_cell

        input = trg[0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs

class ChatbotResponder:
    def __init__(self, model_path, data_path, device):
        self.device = device
        with open(data_path, "rb") as file:
            preprocessed_data = pickle.load(file)
        self.tokenizer = preprocessed_data['tokenizer']
        self.input_dim = len(self.tokenizer.word_index) + 1
        self.output_dim = len(self.tokenizer.word_index) + 1
        self.PAD_IDX = self.tokenizer.word_index['<pad>']

        # Model Initialization
        EMB_DIM = 256
        HID_DIM = 512
        N_LAYERS = 2
        DROPOUT = 0.5
        self.enc = Encoder(self.input_dim, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT).to(self.device)
        self.dec = Decoder(self.output_dim, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT).to(self.device)
        self.model = Seq2Seq(self.enc, self.dec, self.device).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def _tokenize(self, sentence):
        tokens = self.tokenizer.texts_to_sequences([sentence])
        return torch.tensor(tokens).to(self.device)

    def _decode_tensor(self, tensor):
        decoded_words = []
        for tok_id in tensor:
            decoded_words.append(self.tokenizer.index_word[tok_id.item()])
            if tok_id.item() == self.tokenizer.word_index['<end>']:
                break
        return ' '.join(decoded_words[1:])

    def generate_response(self, user_input):
        src_tensor = self._tokenize(user_input)
        with torch.no_grad():
            hidden, cell = self.enc(src_tensor)
            trg_indexes = [self.tokenizer.word_index['<start>']]
            for i in range(100):
                trg_tensor = torch.tensor([trg_indexes[-1]]).to(self.device)
                output, hidden, cell = self.dec(trg_tensor, hidden, cell)
                pred_token = output.argmax(1).item()
                trg_indexes.append(pred_token)
                if pred_token == self.tokenizer.word_index['<end>']:
                    break
        trg_tokens = torch.tensor(trg_indexes, dtype=torch.long)
        response = self._decode_tensor(trg_tokens)
        return response

MODEL_PATH = "/Users/vishalpatel/Desktop/Data Science/Chatbot/data_folder/test18/HotelAssist/immigration/HotelAssist/Final/HotelAssist/chatbot_model_best.pth"
# BASE_PATH = "HotelAssist/"
DATA_PATH = "/Users/vishalpatel/Desktop/Data Science/Chatbot/data_folder/test18/HotelAssist/immigration/HotelAssist/Final/HotelAssist/preprocessed_data.pkl"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
responder = ChatbotResponder(MODEL_PATH, DATA_PATH, DEVICE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        content = request.json
        user_input = content['user_input']
        response = responder.generate_response(user_input)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)})

