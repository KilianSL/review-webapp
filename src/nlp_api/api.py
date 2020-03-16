import torch
import torch.nn.functional as F 
from model import BERTGRUSentiment
from utils import predict_sentiment, tokenize_sentence

model = BERTGRUSentiment(256,1,2,True,0.25)
model_path = 'src/nlp_api/models/BERTGRU_model.pth'
model.load_state_dict(torch.load(model_path))
model.eval()



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
    score = predict_sentiment(model, tokenize_sentence, review)
    print(score)
    return jsonify(float(score))


app.run()
