
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from PIL import Image
import numpy as np
import json
import os

# Function to create the model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(39, activation='softmax')
    ])
    return model

# Function to load model weights
def load_weights(model, weights_path):
    model.load_weights(weights_path)

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to make predictions
def predict_image(model, preprocessed_image):
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction, axis=1)
    class_labels = [
        "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
        "Background_without_leaves", "Blueberry___healthy", "Cherry___Powdery_mildew", "Cherry___healthy",
        "Corn___Cercospora_leaf_spot Gray_leaf_spot", "Corn___Common_rust", "Corn___Northern_Leaf_Blight", "Corn___healthy",
        "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
        "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy", "Pepper,_bell___Bacterial_spot",
        "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
        "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew", "Strawberry___Leaf_scorch",
        "Strawberry___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
        "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites_Two-spotted_spider_mite",
        "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
    ]
    predicted_label = class_labels[predicted_class[0]]
    return predicted_label

# Function to load recommendations
def load_recommendations(json_path):
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            recommendations = json.load(f)
        return recommendations
    else:
        st.error(f"JSON file not found at {json_path}")
        return None

# Function to get recommendation based on prediction
def get_recommendation(predicted_label, recommendations):
    return recommendations.get(predicted_label, {"Description": "No recommendations available for this class."})

# Function to load accuracy score
def load_accuracy_score(score_path):
    if os.path.exists(score_path):
        with open(score_path, 'r') as f:
            accuracy_score = f.read().strip()
        return accuracy_score
    else:
        st.error(f"Accuracy score file not found at {score_path}")
        return None

st.title("Diagnostic Tool")

# Create columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Load the model and weights
        model = create_model()
        load_weights(model, './plant_disease_model.keras')
        #load_weights(model, './model.weight.h5')

        # Preprocess the image and make a prediction
        preprocessed_image = preprocess_image(image)
        predicted_label = predict_image(model, preprocessed_image)

        # Load recommendations
        recommendations = load_recommendations('./PlantVillageRecommendations.json')
        recommendation = get_recommendation(predicted_label, recommendations)



with col2:
    if uploaded_file is not None:
        st.subheader("Results")
        st.markdown(f"**Predicted Class:** {predicted_label}")

        st.markdown("### Recommendations")
        st.markdown(f"**Description:** {recommendation['Description']}")
        st.markdown(f"**Diagnosis:** {recommendation['Diagnosis']}")
        st.markdown(f"**Recommended Action:** {recommendation['Recommended_Action']}")
        st.markdown(f"**Treatment:** {recommendation['Treatment']}")
        st.markdown(f"**Treatment Timing:** {recommendation['Treatment_Timing']}")
        st.markdown(f"**Prevention:** {recommendation['Prevention']}")
        st.markdown(f"**Monitoring:** {recommendation['Monitoring']}")
        st.markdown(f"**Cultural Practices:** {recommendation['Cultural_Practices']}")
        st.markdown(f"**Physical Controls:** {recommendation['Physical_Controls']}")
        st.markdown(f"**Biological Controls:** {recommendation['Biological_Controls']}")


