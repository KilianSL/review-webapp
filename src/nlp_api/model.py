import torch
import torch.nn as nn
from transformers import BertModel

class BERTGRUSentiment(nn.Module): # Model to predict sentiment using pretrained BERT embeddings + Gated Recurrent Unit
    def __init__(self, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()

        self.bert = BertModel.from_pretrained('bert-best-uncased')

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
