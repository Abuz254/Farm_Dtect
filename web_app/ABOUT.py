import streamlit as st

# Title and content without custom CSS
st.title("Plant Disease Detection & Recommendations")

# About Us section
st.header("Welcome to the Plant Disease Detection and Recommendation System")

st.write("""
This system is designed to assist farmers, gardeners, and plant enthusiasts in identifying and managing plant diseases.
Leveraging state-of-the-art deep learning models trained on extensive datasets, the system offers precise disease detection and tailored recommendations.
Our goal is to enhance plant health and productivity through early diagnosis and effective treatments.
""")

#st.image("https://miro.medium.com/v2/resize:fit:600/1*dQ74HRUH5ot3oRpS-f6DVQ.jpeg", caption="Plant Disease Detection Tool", use_column_width=True)


# Key Features section
st.subheader("Key Features")
st.write("""
- **Accurate Disease Detection:** Uses advanced machine learning algorithms to identify plant diseases from images.
- **Detailed Recommendations:** Provides actionable recommendations based on the detected disease.
- **User-Friendly Interface:** Easy-to-navigate web interface for quick access and efficient use.
- **Regular Updates:** Continuously updated with new data and recommendations to ensure accuracy and relevance.
""")

# Contact Information section
st.subheader("Contact Us")
st.write("""
- **Email:** [support@farmdetect.com](mailto:support@farmdetect.com)
- **Phone:** (123) 456-7890
- **Website:** [www.farmdetect.com](http://www.farmdetect.com)
""")


if st.button('Learn More'):
    st.write("Thank you for your interest! We'll get back to you soon.")
