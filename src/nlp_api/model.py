import torch
import torch.nn as nn
from transformers import BertModel
from transformers import BertTokenizer

class Tokenizer():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_input_len = self.tokenizer.max_model_input_sizes['bert-base-uncased']
        self.init_token_idx = self.tokenizer.cls_token_id
        self.eos_token_idx = self.tokenizer.sep_token_id
        self.pad_token_idx = self.tokenizer.pad_token_id
        self.unk_token_idx = self.tokenizer.unk_token_id

    def tokenize_sentence(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        tokens = tokens[:self.max_input_len-2]
        return tokens


class BERTGRUSentiment(nn.Module): # Model to predict sentiment using pretrained BERT embeddings + Gated Recurrent Unit
    def __init__(self, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        embedding_dim = self.bert.config.to_dict()['hidden_size']

        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, batch_first=True, dropout=dropout)

        if bidirectional:
            fc_dim = hidden_dim * 2
        else:
            fc_dim = hidden_dim

        self.out = nn.Linear(fc_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # B=batch_size, T=tokens in x, 
        # x = [B,T]

        with torch.no_grad():
            embedded = self.bert(x)[0]
        # embedded = [B,T,embedding_len]

        _, hidden = self.rnn(embedded)
        # hidden = [n_layers*n_direction, B, embedding_len]

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        # hidden = [B,hidden_dim]

        output = self.out(hidden)
        
        return output
