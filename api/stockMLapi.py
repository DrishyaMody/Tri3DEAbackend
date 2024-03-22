from flask import Flask, request, jsonify, Blueprint
import pandas as pd
from model.stockMLmodel import *

# Initialize a new web application using Flask.
app = Flask(__name__)

# Create a blueprint for our API. Think of this as a component of the app that handles all 'stock' related operations.
stock_api = Blueprint('stock_api', __name__, url_prefix='/api/stock')

@stock_api.route('/predict', methods=['POST'])
def predict_market_cap():
    try:
        # Extract the data sent by the user.
        data = request.json
        # Get the 'Last Sale' and 'Volume' values from the data, or use 0 as a default.
        last_sale = float(data.get('Last Sale', 0))
        volume = float(data.get('Volume', 0))
        
        # Check if volume = 0 and change market cap to match
        if volume == 0:
            # Create a response object with a market cap = 0.
            response = {'predicted_market_cap': '0.00'}
            return jsonify(response), 200
        
        # If volume is not 0, prepare the data for the model.
        features = [[last_sale, volume]]
        
        # Use the model to predict the market cap based on the provided features.
        predicted_market_cap = rf_regressor.predict(features)[0]
        
        # Round the prediction to two decimal places for easier reading.
        formatted_market_cap = f"{predicted_market_cap:.2f}"
        
        # Create a response object containing the prediction.
        response = {'predicted_market_cap': formatted_market_cap}
        return jsonify(response), 200
    except Exception as e:
        # If something goes wrong, send back an error message with status code 400, which means 'Bad Request'.
        return jsonify({'error': str(e)}), 400

# Register the blueprint with the main app.
app.register_blueprint(stock_api)

if __name__ == '__main__':
    app.run(debug=True)
