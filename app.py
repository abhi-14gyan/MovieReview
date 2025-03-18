from flask import Flask, request, jsonify
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from flask_cors import CORS  # ✅ Allow cross-origin requests (fixes CORS issues)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # ✅ Enables Cross-Origin Resource Sharing

# Load trained model, tokenizer, and label encoder
model = load_model("sentiment_model.h5")

with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

with open("label_encoder.pkl", "rb") as handle:
    label_encoder = pickle.load(handle)

# Define constants
MAX_LEN = 200  # Ensure it matches the max length used during training

# Prediction function
def predict_sentiment(review):
    review = review.lower()
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN)
    prediction = model.predict(padded_sequence)
    
    predicted_label = int(np.argmax(prediction))  # ✅ Convert np.int64 to Python int
    sentiment = label_encoder.inverse_transform([predicted_label])[0]  # ✅ Extract string
    
    return sentiment  # Now it returns a proper string

# Flask route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        review = data.get("review", "")
        
        if not review:
            return jsonify({"error": "No review provided"}), 400
        
        sentiment = predict_sentiment(review)
        return jsonify({"review": review, "sentiment": str(sentiment)})  # ✅ Convert to string

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
