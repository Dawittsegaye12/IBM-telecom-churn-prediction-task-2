from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging

app = Flask(__name__)

# Load the saved model
model = joblib.load('model/best_model.pkl')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log the incoming request
        logging.info('Received prediction request')

        # Get JSON data from the request
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)
        probability = model.predict_proba(features).max()

        # Log the prediction result
        logging.info(f'Prediction: {prediction[0]}, Probability: {probability}')

        return jsonify({
            'prediction': int(prediction[0]),
            'probability': float(probability)
        })
    except Exception as e:
        logging.error(f'Error during prediction: {e}')
        return jsonify({'error': str(e)})

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        # Log the incoming batch request
        logging.info('Received batch prediction request')

        # Get JSON data from the request
        data = request.get_json()
        features_list = np.array(data['features'])

        # Make batch predictions
        predictions = model.predict(features_list).tolist()
        probabilities = model.predict_proba(features_list).max(axis=1).tolist()

        # Log the batch prediction result
        logging.info(f'Batch Predictions: {predictions}')

        return jsonify({
            'predictions': predictions,
            'probabilities': probabilities
        })
    except Exception as e:
        logging.error(f'Error during batch prediction: {e}')
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)