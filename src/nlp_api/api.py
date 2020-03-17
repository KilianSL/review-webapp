import torch
import torch.nn.functional as F
import pytorch_pretrained_bert as ppb
from model import BERTGRUSentiment, Tokenizer
from utils import predict_sentiment

model = BERTGRUSentiment(256,1,2,True,0.25)
#model_path = 'src/nlp_api/models/BERTGRU_model.pth'
#model.load_state_dict(torch.load(model_path))
model.eval()
tokenizer = Tokenizer()


import flask
from flask import request
from flask import jsonify
from flask_cors import CORS

app = flask.Flask(__name__)
app.config["DEBUG"] = True
CORS(app)


@app.route('/', methods=['GET']) # Returns sentiment score based on value of get request
def predict():
    review = request.args['value']
    # print(review)
    score = predict_sentiment(model, tokenizer, review)
    print(score)
    return jsonify(float(score))


app.run()
