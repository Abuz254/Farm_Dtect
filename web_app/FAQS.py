import streamlit as st

# Set page configuration for white background and black text
st.title("Frequently Asked Questions")

# Apply custom CSS for the background and text color
st.markdown(
    """
    <style>
    body {
        background-color: white;
        color: black;
    }
    .st-expander {
        background-color: #f9f9f9;
    }
    .st-expander .st-expanderHeader {
        color: black;
    }
    .st-expanderContent p {
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Frequently Asked Questions")

# Define FAQ questions and answers
faqs = [
    {
        "question": "How accurate is the disease detection?",
        "answer": "The model is trained on a large dataset and has a high accuracy. However, it is always recommended to consult a professional for a more accurate diagnosis."
    },
    {
        "question": "Can I use this tool for any type of plant?",
        "answer": "The tool is currently trained on specific plant diseases. Please refer to the list of supported plants for more information."
    },
    {
        "question": "How do I upload an image?",
        "answer": "Navigate to the Diagnostic Tool section in the sidebar, and you will find an option to upload an image."
    },
    {
        "question": "Is my data secure with this tool?",
        "answer": "Yes, we take data privacy seriously. All uploaded images and data are processed securely and are not shared with third parties."
    },
    {
        "question": "How often is the tool updated?",
        "answer": "The tool is regularly updated with new data and improvements to enhance accuracy and provide the latest recommendations."
    },
    {
        "question": "Can I download the results from the diagnostic tool?",
        "answer": "Currently, the tool does not support downloading results directly. However, you can take screenshots or manually record the information."
    },
    {
        "question": "What should I do if the tool doesn't recognize my plant disease?",
        "answer": "If the tool doesn't recognize your plant disease, please consult a professional for a more accurate diagnosis. You can also provide feedback to help improve the tool."
    }
]

# Display FAQs with expandable sections
for faq in faqs:
    with st.expander(faq["question"]):
        st.write(faq["answer"])
