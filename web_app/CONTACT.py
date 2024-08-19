import streamlit as st

# Title
st.title("Plant Disease Detection & Recommendations")

# Header
st.header("Welcome")

# Description
st.write("""
This system helps in identifying plant diseases and provides tailored recommendations.
""")

# Display Image
st.image("https://miro.medium.com/v2/resize:fit:600/1*dQ74HRUH5ot3oRpS-f6DVQ.jpeg", caption="Plant Disease Detection Tool", use_column_width=True)


# Key Features


# Contact Form
st.subheader("Contact Form")
email = st.text_input("Your Email")
message = st.text_area("Your Message")



if st.button("Send Message"):
    if email and message:
        st.success("Message sent! We'll get back to you soon.")
    else:
        st.error("Please provide both email and message.")

# Contact Information
st.subheader("Contact Us")
st.write("""
- **Email:** [support@farmdetect.com](mailto:support@farmdetect.com)
- **Phone:** (123) 456-7890
- **Website:** [www.farmdetect.com](http://www.farmdetect.com)
""")
