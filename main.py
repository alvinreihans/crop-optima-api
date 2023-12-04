import os
import io
import IPython
from IPython import get_ipython
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify

model = keras.models.load_model("model.h5")

app = Flask(__name__)

def predict_crop(N, P, K, ph):
    input_data = np.array([[N, P, K, ph]])
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction)
    crop_label = label_encoder.inverse_transform([predicted_class])[0]
    return crop_label

@app.route('/', methods=['POST'])
def root():
    data = request.get_json()
    n = data.get('n')
    p = data.get('p')
    k = data.get('k')
    ph = data.get('ph')
    result = predict_crop(n, p, k, ph);
    return jsonify({'status': '200 OK', 'data': result})

if __name__ == "__main__":
    app.run(debug=True)