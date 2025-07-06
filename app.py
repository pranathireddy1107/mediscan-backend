from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

# Load trained model
with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    vectorizer = model_data['vectorizer']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', '')
        X_input = vectorizer.transform([symptoms])
        prediction = model.predict(X_input)
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'prediction': 'Error occurred'}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)
