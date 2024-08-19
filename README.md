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

### NASA Earth Observing System Data and Information System (EOSDIS)

- **Source:** [NASA EOSDIS](https://earthdata.nasa.gov/)
- **Columns:**
  - `date`: Date of the observation
  - `latitude`: Latitude of the observation location
  - `longitude`: Longitude of the observation location
  - `temperature`: Surface temperature (in Celsius)
  - `precipitation`: Precipitation levels (in mm)
  - `humidity`: Relative humidity (in %)
  - `solar_radiation`: Solar radiation levels (in W/m²)
  - `soil_moisture`: Soil moisture levels (in m³/m³)

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

1. Start

    User Accesses Web Application

2. Homepage

    Action: User sees options such as "Upload Image" and "View Recommendations."
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

7. User Actions

    Option 1: User can choose to upload another image.
    Option 2: User can view more information or additional resources.
    Option 3: User can contact support if needed.

8. End

    Action: User completes their interaction.

Here is a visual representation of the user flow:

+----------------------------+
|  Start                     |
|  User accesses web app     |
+----------------------------+
             |
             V
+----------------------------+
|  Homepage                  |
|  - Options: Upload Image    |
|  - View Recommendations    |
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
|  User Actions              |
|  - Upload another image    |
|  - View more information   |
|  - Contact support         |
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



## Screenshots

## Screenshots

### Home Page
![Home Page Screenshot](./screenshots/home_page.png)

### Dashboard
![Dashboard Screenshot](./screenshots/dashboard.png)

### User Profile
![User Profile Screenshot](./screenshots/user_profile.png)



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
- **Concepts Applied:** 
  - **Feature Engineering:** Used advanced feature engineering techniques to improve model performance.
  - **Ensemble Methods:** Applied ensemble methods like stacking and boosting to enhance predictive accuracy.
- **Domain-Specific Insights:**
  - Integrated knowledge of climate impacts on agriculture to tailor the model’s features and predictions.

### Integration of Additional Python Packages
- **Packages Used:**
  - **XGBoost:** Applied for gradient boosting due to its superior performance on the dataset.
  - **TensorFlow:** Used for deep learning tasks to handle complex patterns in the data.
- **Implementation Details:**
  ```python
  import xgboost as xgb
  model = xgb.XGBClassifier()
  model.fit(X_train, y_train)


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
