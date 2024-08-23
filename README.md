# Data-Driven Crop Diagnostic Platform for Farmers

## Project Overview

The Data-Driven Crop Diagnostic Platform is an innovative AI-powered solution designed to help rural farmers diagnose crop diseases and optimize agricultural practices. By leveraging advanced image classification technology and integrating environmental data, this platform provides actionable insights and recommendations to enhance crop health and productivity.

## Objectives

- **Identify Crop Diseases:** Utilize image classification to accurately identify various crop diseases.
- **Predictive Model Development:** Develop a model that combines image data with environmental factors to provide comprehensive insights into crop health.
- **Actionable Recommendations:** Offer practical advice for managing and preventing crop diseases based on model predictions.
- **Time Series Analysis:** Explore time series analysis to predict the spread of crop diseases influenced by environmental conditions.

## Solution Overview

Our platform features a user-friendly mobile application that allows farmers to upload images of their crops. The AI algorithms analyze these images to diagnose issues such as pests, diseases, and nutrient deficiencies. The platform also incorporates environmental data to provide a comprehensive view of crop health and offer tailored recommendations.

### Key Features

- **Instant Diagnostics:** Real-time analysis of crop images with high accuracy.
- **Predictive Insights:** Integration of image and environmental data for comprehensive health insights.
- **Actionable Recommendations:** Tailored advice on managing and preventing crop diseases.
- **Support Network:** Access to community forums, real-time chat support, and local field agents.

## Data Sources

### PlantVillage Dataset

- **Source:** [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Columns:**
  - `image_id`: Unique identifier for each image
  - `image_path`: Path to the image file
  - `crop_type`: Type of crop (e.g., apple, grape, tomato)
  - `disease_type`: Type of disease (e.g., apple scab, grape black rot)
  - `health_status`: Health status of the plant (healthy or diseased)
  - `image`: The image data itself

### Mendeley Data Repository

- **Source:** [Mendeley Data Repository](https://data.mendeley.com/)
- **Columns:**
  - `image_id`: Unique identifier for each image
  - `image_path`: Path to the image file
  - `crop_type`: Type of crop
  - `disease_type`: Type of disease or pest
  - `nutrient_deficiency`: Type of nutrient deficiency (if applicable)
  - `location`: Geographic location where the image was taken
  - `image`: The image data itself



## Data Preparation

### 1. Data Loading

Load images from specified folders using a data generator:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to your image folders
no_aug_dir = 'Plant_leave_diseases_dataset_without_augmentation'
aug_dir = 'Plant_leave_diseases_dataset_with_augmentation'

# Data generators for loading images
no_aug_datagen = ImageDataGenerator(rescale=1./255)
aug_datagen = ImageDataGenerator(rescale=1./255)

no_aug_generator = no_aug_datagen.flow_from_directory(
    no_aug_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

aug_generator = aug_datagen.flow_from_directory(
    aug_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)




## Tech Stack



**Client:** Flutter, Firebase, Dart, HTML, CSS, JavaScript

**Server:** Python, FastAPI, TensorFlow

**Database:** Firebase Firestore

**AI/ML:** TensorFlow, Keras, OpenCV

**Web Application:** Java, HTML, CSS

**Cloud Services:** Google Cloud, Firebase Hosting

**DevOps:** Docker, GitHub Actions, Firebase CI/CD

**Version Control:** Git, GitHub
## Flowchart
Flowchart for the Project

+--------------------------------------+
| 1. Project Overview & Narrative       |
+--------------------------------------+
                   |
                   v
+--------------------------------------+
| 2. Data Collection & Preparation      |
+--------------------------------------+
                   |
                   v
+--------------------------------------+
| 3. Model Selection & Development      |
+--------------------------------------+
                   |
                   v
+--------------------------------------+
| 4. Web Application Development        |
|  - Design UI/UX                       |
|  - Implement Image Upload Feature     |
|  - Integrate Model with Backend       |
|  - Display Results & Recommendations  |
+--------------------------------------+
                   |
                   v
+--------------------------------------+
| 5. Model Interpretation & Rationale   |
+--------------------------------------+
                   |
                   v
+--------------------------------------+
| 6. Critical Evaluation                |
+--------------------------------------+
                   |
                   v
+--------------------------------------+
| 7. Documentation & Reporting           |
+--------------------------------------+



## User Flowchart for the Web Application

User Flowchart for the Web Application with Voice-over Integration

1. Start
    Action: User accesses the web application.

2. Homepage
        Action: User sees options such as "Upload Image," "View Recommendations," and "Voice-over Language Options."
        Decision: User chooses an option.

3. Upload Image
        Action: User clicks "Upload Image."
        Form: User is prompted to upload an image of their crop.
        Decision: User selects an image file and submits.

4. Image Upload
        Action: Image is uploaded to the server.
        Processing: The web application sends the image to the model for analysis.

5. Model Analysis
        Action: Model analyzes the image and generates a diagnosis.
6. Decision: The model determines the crop’s condition and provides recommendations.

7. View Diagnosis and Recommendations
        Action: User receives a diagnosis and recommendations based on the image analysis.
8. Output: Display results on the screen with details such as crop health status and improvement suggestions.

9. Voice-over Interpretation
        Action: User can select a language for voice-over interpretation.
10. Processing: The web application uses a voice-over application to read out the diagnosis and recommendations in the selected language.

11. User Actions
        Option 1: User can choose to upload another image.
        Option 2: User can view more information or additional resources.
        Option 3: User can contact support if needed.
        Option 4: User can choose to hear the diagnosis and recommendations in a different language.

12. End
        Action: User completes their interaction.

Visual Representation of the User Flow

+----------------------------+
|  Start                     |
|  User accesses web app     |
+----------------------------+
             |
             V
+----------------------------+
|  Homepage                  |
|  - Options:                 |
|    * Upload Image           |
|    * View Recommendations   |
|    * Voice-over Language Options |
+----------------------------+
             |
             V
+----------------------------+
|  Upload Image              |
|  - Prompt to upload image  |
+----------------------------+
             |
             V
+----------------------------+
|  Image Upload              |
|  - Send image to server    |
+----------------------------+
             |
             V
+----------------------------+
|  Model Analysis            |
|  - Analyze image           |
|  - Generate diagnosis      |
+----------------------------+
             |
             V
+----------------------------+
|  View Diagnosis            |
|  - Display results         |
|  - Recommendations         |
+----------------------------+
             |
             V
+----------------------------+
|  Voice-over Interpretation |
|  - Select language         |
|  - Read out diagnosis and  |
|    recommendations         |
+----------------------------+
             |
             V
+----------------------------+
|  User Actions              |
|  - Upload another image    |
|  - View more information   |
|  - Contact support         |
|  - Change voice-over language |
+----------------------------+
             |
             V
+----------------------------+
|  End                       |
|  User completes interaction |
+----------------------------+

## Demo


<img src="path/to/your/demo.gif" alt="Project Demo">

<p>Or you can view the live demo <a href="https://your-demo-link.com">here</a>.</p>



## Documentation

Project Overview

This project aims to build a data-driven farm detection model that helps identify and diagnose issues with crops based on images. The model provides recommendations for improving crop health and productivity. The project includes a web application where farmers can post images and receive diagnoses or recommendations.
Environment Setup
Platform Information

    OS: Ubuntu 22.04 or equivalent
    Python Version: 3.9

Dependencies

Ensure that you have all required software and libraries installed. Use the following command to install dependencies from requirements.txt:

# Create a virtual environment (optional)
python -m venv env

# Activate the virtual environment (Windows)
.\env\Scripts\activate

# Activate the virtual environment (macOS/Linux)
source env/bin/activate

# Install dependencies
pip install -r requirements.txt


'requirements.txt':
numpy==1.24.0
pandas==2.1.0
scikit-learn==1.2.0
matplotlib==3.7.0
tensorflow==2.13.0
flask==2.3.2


Data Sources
Data Description

    Crop Health Images Dataset
        Source: [Link to dataset]
        Format: Images in JPEG/PNG format
        Description: Contains images of crops for classification and diagnosis.

    Climate Data
        Source: [Link to dataset]
        Format: CSV
        Description: Contains climate data that may affect crop health.

**Data Access**

    - Download the Crop Health Images Dataset from [kaggle.com].
    - Download images from  **Source:** [Mendeley Data Repository](https://data.mendeley.com/)
    - Download the Climate Data from **Source:** [NASA EOSDIS](https://earthdata.nasa.gov/).
    - Place the downloaded files in the data/ directory.

**Setup Instructions**
Installation Steps

   1.  Clone the repository:
   git clonegit@github.com:Abuz254/Farm_Dtect.git
cd data-driven-farm-detection

  2. Install dependencies:
  pip install -r requirements.txt

  3. Set up environment variables:
  export FLASK_APP=app.py

**Running the Project**
Execution Instructions
1. Run the Web Application
flask run
This will start a local server on http://127.0.0.1:5000 where you can access the web application.

2. Run the Model Training Script:
python train_model.py --data data/crop_health_images --output models/crop_model.h5

Usage Examples

    Using the Web Application:
        Open your browser and navigate to http://127.0.0.1:5000.
        Upload a crop image to receive a diagnosis or recommendation.

    Training the model:
      python train_model.py --data data/crop_health_images --output models/crop_model.h5


Troubleshooting

    Missing Dependencies: Ensure all packages listed in requirements.txt are installed.
    Data Not Found: Verify that the data files are correctly placed in the data/ directory.
    Flask Server Issues: Ensure that the Flask server is running and the FLASK_APP environment variable is correctly set.

Additional Notes

     The code is commented to explain key functions and logic.
     For further information, refer to the Project Documentation.


## Deployment

To deploy this project run

## Deployment

### Overview

This project is deployed using the following components:

- **Web Application:** Built with Streamlit for a streamlined and interactive user interface for ML and AI 
- **API Server:** Hosted on Flask.
- **Model:** Deployed on Streamlit Community Cloud..

### Installation

To set up the environment, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone git@github.com:Abuz254/Farm_Dtect.git

2. **Install dependencies**
   ```bash
   cd project-directory
npm install

3. **Start application**
    ```bash
    npm start




## Advanced Learning and Intergration


### Advanced Domain Knowledge

    Concepts Applied:
        Feature Engineering: Implemented sophisticated feature engineering techniques to enhance model performance, including creating interaction features and applying domain-specific transformations.
        Ensemble Methods: Leveraged advanced ensemble methods such as stacking and boosting to improve predictive accuracy and robustness of the model.
    Domain-Specific Insights:
        Utilized expertise in the effects of climate change on agriculture to inform the selection and engineering of features, ensuring the model's predictions are relevant and actionable for agricultural applications.

Integration of Additional Python Packages

    Packages Used:
        XGBoost: Employed XGBoost for its high performance in handling complex datasets and delivering robust gradient boosting results.
        TensorFlow: Utilized TensorFlow for deep learning tasks to capture intricate patterns and relationships within the data.
    Implementation Details:
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# XGBoost Model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)

# TensorFlow Model
tf_model = Sequential()
tf_model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
tf_model.add(Dense(32, activation='relu'))
tf_model.add(Dense(1, activation='sigmoid'))
tf_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
tf_model.fit(X_train, y_train, epochs=10, batch_size=32)

Web Application with Voice-Over Functionality

User Flowchart for the Web Application

1. Start
        User accesses the web application.

2. Homepage
        Action: User sees options such as "Upload Image," "View Recommendations," and "Listen to Diagnosis."
        Decision: User chooses an option.

3. Upload Image
        Action: User clicks "Upload Image."
        Form: User is prompted to upload an image of their crop.
        Decision: User selects an image file and submits.

4. Image Upload
        Action: Image is uploaded to the server.
        Processing: The web application sends the image to the model for analysis.

5. Model Analysis
        Action: Model analyzes the image and generates a diagnosis.
        Decision: The model determines the crop’s condition and provides recommendations.

6. View Diagnosis and Recommendations
        Action: User receives a diagnosis and recommendations based on the image analysis.
        Output: Display results on the screen with details such as crop health status and improvement suggestions.

7. Voice-Over Interpretation
        Action: User can choose to listen to the diagnosis and recommendations.
        Integration: The web application uses a text-to-speech (TTS) engine to provide the diagnosis and recommendations in multiple languages.
        Implementation:

from gtts import gTTS
import pyttsx3

def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    tts.save("diagnosis.mp3")
    # Play the saved audio file
    # Code to play the audio file (implementation depends on the platform)

8. User Actions

    Option 1: User can choose to upload another image.
    Option 2: User can view more information or additional resources.
    Option 3: User can contact support if needed.

9. End

    Action: User completes their interaction.

Visual Representation of the User Flow

+----------------------------+
| Start |
| User accesses web app |
+----------------------------+
|
V
+----------------------------+
| Homepage |
| - Options: Upload Image, |
| View Recommendations, |
| Listen to Diagnosis |
+----------------------------+
|
V
+----------------------------+
| Upload Image |
| - Prompt to upload image |
+----------------------------+
|
V
+----------------------------+
| Image Upload |
| - Send image to server |
+----------------------------+
|
V
+----------------------------+
| Model Analysis |
| - Analyze image |
| - Generate diagnosis |
+----------------------------+
|
V
+----------------------------+
| View Diagnosis |
| - Display results |
| - Recommendations |
+----------------------------+
|
V
+----------------------------+
| Voice-Over Interpretation |
| - Text-to-Speech in |
| multiple languages |
+----------------------------+
|
V
+----------------------------+
| User Actions |
| - Upload another image |
| - View more information |
| - Contact support |
+----------------------------+
|
V
+----------------------------+
| End |
| User completes interaction |
+----------------------------+

This flow incorporates the voice-over feature, allowing users to receive spoken diagnoses and recommendations in multiple languages, enhancing accessibility and user experience.


## Potential Clients

This project can be utilized by the following types of companies:

- **Agriculture Technology Companies**: For improving crop management and yield prediction using AI and machine learning.
- **Environmental Organizations**: To monitor and assess the impact of climate change on agriculture in different regions.
- **Government Agencies**: For policy-making and implementing sustainable farming practices.
- **Research Institutions**: Conducting studies on climate change and agricultural productivity.
- **Educational Institutions**: Incorporating this project into academic programs focused on environmental science and data analytics.



## Acknowledgements

I would like to express my sincere gratitude to the following individuals and organizations for their invaluable support and contributions to this project:

**Kaggle.com:** For providing access to the PlantVillage dataset, which was instrumental in training and testing the image classification models used in this project.

**Ms.Asha Deen:** For offering guidance, insightful feedback, and encouragement throughout the development of this project. Your expertise in machine learning and data science was critical in shaping the direction of this work.

**Team Members**
claudia.sagini@student.moringaschool.com
ian.kiptoo@student.moringaschool.com
allan.kiplagat@student.moringaschool.com
samuel.wanga@student.moringaschool.com
jerry.narkiso@student.moringaschool.com
prossy.nansubuga@student.moringaschool.com: For your collaboration, hard work, and dedication. Your contributions to data preprocessing, model development, and testing were essential in achieving the project’s objectives.
  
Open-Source Community: For the wealth of knowledge, tools, and resources available through platforms like GitHub, TensorFlow, and PyTorch, which made this project possible.

Rural Farmers in Kenya: For inspiring the purpose of this project. Your experiences and challenges motivated the creation of this platform aimed at improving agricultural practices and crop health diagnosis.

Thank you all for your contributions and support, which made this project a success.

## Recommendations for Scaling the Farm Detection Project

1. Transition to FastAPI

Advantages:

    High performance with asynchronous capabilities.
    Automatic interactive API documentation.
    Enhanced scalability for handling concurrent requests.

Action Steps:

    Replace Flask with FastAPI for developing the web application and API endpoints.
    Implement endpoints for image uploads, predictions, and user management.
    Utilize FastAPI's features for data validation and serialization to streamline development.

2. Develop a Mobile Application

Advantages:

    Increases accessibility and user engagement.
    Provides real-time interaction through push notifications and instant feedback.

Action Steps:

    Develop a mobile app using Flutter or React Native for cross-platform support.
    Integrate the FastAPI backend with the mobile app for seamless image uploads and predictions.
    Include features for capturing images, viewing past diagnoses, and accessing recommendations.

3. Expand Data Collection

Advantages:

    Enhances model accuracy and generalization with a larger, more diverse dataset.
    Allows for more comprehensive analysis of crop diseases.

Action Steps:

    Implement a crowdsourcing strategy or collaborate with farmers to collect a diverse set of crop images.
    Apply data augmentation techniques to artificially expand the dataset.
    Develop a feature in the mobile app or web application for users to submit images and associated metadata.

4. Implement Geographic Clustering of Diseases

Advantages:

    Provides insights into disease distribution across different regions.
    Helps in targeted resource allocation and localized recommendations.

Action Steps:

    Collect geographical data alongside disease images to analyze disease patterns.
    Use clustering algorithms (e.g., K-means, DBSCAN) to group diseases based on their geographical occurrence.
    Create visualizations such as heatmaps or geographic plots to display disease distribution and trends.

5. Enhance Model and Application Integration

Advantages:

    Improved efficiency and user experience through robust backend and frontend integration.
    Facilitates continuous improvement and adaptability of the system.

Action Steps:

    Regularly update and retrain the model with new data to improve accuracy and adapt to emerging diseases.
    Continuously test and refine the FastAPI backend and mobile application based on user feedback and performance metrics.
    Ensure smooth integration between the web/mobile applications and the backend model for seamless user interaction.