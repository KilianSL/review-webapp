import torch
from transformers import BertTokenizer

def predict_sentiment(model, tokenizer, sentence):
    model.eval()
    tokens = tokenizer.tokenize_sentence(sentence)
    tokens = tokens[:tokenizer.max_input_len-2]
    indexed = [tokenizer.init_token_idx] + tokenizer.tokenizer.convert_tokens_to_ids(tokens) + [tokenizer.eos_token_idx]
    tensor = torch.LongTensor(indexed)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()