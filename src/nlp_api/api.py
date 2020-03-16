import torch
import torch.nn.functional as F 
#from model import CharacterLevelCNN
#from utils import predict_sentiment

#model = CharacterLevelCNN()
#model_path = 'src/nlp_api/models/nlp_model.pth'
#model.load_state_dict(torch.load(model_path))
#model.eval()



import flask
from flask import request
from flask import jsonify
from flask_cors import CORS

app = flask.Flask(__name__)
app.config["DEBUG"] = True
CORS(app)


@app.route('/', methods=['GET']) # Returns sentiment score based on value of get request
def predict():
    #params = model.get_model_parameters()
    review = request.args['value']
    # print(review)
    score = #3 predict_sentiment(model, review, **params)
    print(score)
    return jsonify(float(score))


app.run()
