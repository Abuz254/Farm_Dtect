import streamlit as st

# Function to switch pages
def switch_page(page_name):
    st.session_state.current_page = page_name

# Container for the content
with st.container():
    # Title
    st.title("🌿 Welcome to the Farm Detect Disease Detection and Recommendation System 🌿")
    st.image("https://miro.medium.com/v2/resize:fit:600/1*dQ74HRUH5ot3oRpS-f6DVQ.jpeg", caption="Plant Disease Detection Tool", use_column_width=True)

    # Subtitle and descriptive text
    st.subheader("Welcome to our advanced Plant Disease Detection and Recommendation System! 🌱")
    st.write("""
    This tool is designed to help you:
    - **Upload images** of plant leaves.
    - **Get precise disease predictions** from the images.
    - **Receive comprehensive recommendations** for treatment and prevention.

    **How It Works:**
    1. **Navigate** to the **Diagnostic Tool** section.
    2. **Upload** an image of a plant leaf.
    3. **Receive predictions** and actionable recommendations instantly.

    We aim to support you in keeping your plants healthy and thriving. Explore the features and take advantage of our tool to maintain the best care for your plants.

    For any questions or additional help, check out the [FAQs](#) or contact us through the [Contact](#) page.

    **Let's get started and ensure your plants are always in their best shape!** 🌟
    """)

    # Button to start using the diagnostic tool
    if st.button("Go to Diagnostic Tool"):
        st.write("Redirecting you to the Diagnostic Tool...")
        switch_page("diagnostic")  # Set session state to navigate to the diagnostic tool

    # Optional Footer
    st.markdown("© 2024 Plant Disease Detection System. All Rights Reserved.")

# Initialize session state for page navigation
if "current_page" not in st.session_state:
    st.session_state.current_page = "HOME"

# Check the session state and load the appropriate page
if st.session_state.current_page == "DIAGNOSTIC":
    import diagnostic  # Import the diagnostic tool page
    diagnostic.run()
