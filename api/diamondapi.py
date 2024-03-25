from model.diamondmodel import *
from flask import Flask, request, jsonify, Blueprint
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)
diamond_api = Blueprint('diamond_api', __name__, url_prefix='/api/diamond')

# Example function to predict price - in practice, load your model here
def predict_diamond_price(features):
    # Assuming the 'model' is already loaded or defined elsewhere in your code
    return model.predict(features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the POST request
        data = request.get_json(force=True)
        features = pd.DataFrame(data, index=[0])
        
        # Predict
        prediction = predict_diamond_price(features)
        
        # Return the prediction
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
