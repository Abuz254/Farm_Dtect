from flask import Flask, request, jsonify
import pickle
import numpy as np
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    img = Image.open(io.BytesIO(file.read()))
    
    # Example preprocessing step (resize, normalize, etc.)
    img = img.resize((224, 224))  # Resize image if necessary
    img_array = np.array(img) / 255.0  # Normalize
    img_array = img_array.reshape(1, -1)  # Reshape if needed
    
    # Predict using the loaded model
    prediction = model.predict(img_array)
    
    # Convert the prediction to a string or label
    result = np.argmax(prediction, axis=1)[0]  # Example for a classification model
    
    return jsonify({'prediction': str(result)})

if __name__ == '__main__':
    app.run(debug=True)
