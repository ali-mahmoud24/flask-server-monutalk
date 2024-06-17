from flask import Flask, request, jsonify
from flask_cors import CORS
import os

from Recommendation_System.recommendation import get_museums, get_recommendations
from Chat.chat import output_of_to_genai
from Sentiment_Analysis.main import predict_sentiment
# from CNN.model import show_info



app = Flask(__name__)
CORS(app)



# UPLOAD_FOLDER = '/uploads'  # Define the upload folder path
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



# @app.route('/')
# def hello():
#     return 'Hello, this is your Flask API!'


# # CNN

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'image' not in request.files:
#         return 'No file part'

#     file = request.files['image']

#     if file.filename == '':
#         return 'No selected file'

#     if file:
#       file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#       file.save(file_path)

#       name, info, questions = show_info(file_path)
#       return jsonify({'name': name, 'info': info, 'questions': questions})



# SENTIEMENT ANALYSIS

# Send data as json { "review": ['text..............'] }
@app.route('/predict', methods=['POST'])
def predict_review():

  review = request.json['review']

  rating = predict_sentiment(review)

  return jsonify({'rating': rating})



# RECOMONDATION SYSTEM

@app.route('/museum-options', methods=['GET'])
def get_museum_options():

  museums = get_museums()

  return jsonify({'museums': museums})



# Send data as json { "museum": 'text..............' }
@app.route('/recommend', methods=['POST'])
def recommend_museum():

  museum_name = request.json['museum']

  recommend_museums, search_result = get_recommendations(museum_name)

  return jsonify({'museum': search_result ,'museums': recommend_museums})



# CHAT

# Send data as json { "text": 'text..............' }
@app.route('/chat', methods=['POST'])
def chat():

  text = request.json['text']

  output = output_of_to_genai(text)

  return jsonify({'output': output})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)