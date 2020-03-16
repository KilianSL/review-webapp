# Sentiment Analysis React App

## Starting the service
+ Run `api.py` (will listen on 127.0.0.1:5000)
+ Run `npm start` in project directory (will host on localhost:3000)

## Changelog

**`Version 0.0.2`**
+ Implemented callback function to update UI when sentiment score is returned
+ Implemented changing background colour on progress bar depending on score value
+ Fixed issue with API calls lagging behind state updates

**`Version 0.0.1`**
+ Basic static UI created
+ Created Flask API to return sentiment score from NN
+ API calls predict using pre-trained 1D CNN, trained on movie reviews. 
