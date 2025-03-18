from flask import Flask, request, jsonify
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load trained model, tokenizer, and label encoder
model = load_model("sentiment_model.h5")
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)
with open("label_encoder.pkl", "rb") as handle:
    label_encoder = pickle.load(handle)

# Define constants
MAX_LEN = 200  # Same max length used during training

# Prediction function
def predict_sentiment(review):
    review = review.lower()
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN)
    prediction = model.predict(padded_sequence)
    predicted_label = np.argmax(prediction)
    sentiment = label_encoder.inverse_transform([predicted_label])
    return sentiment[0]

# Flask route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        review = data.get("review", "")
        
        if not review:
            return jsonify({"error": "No review provided"}), 400
        
        sentiment = predict_sentiment(review)
        return jsonify({"review": review, "sentiment": sentiment})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
