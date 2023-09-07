from __future__ import print_function

import streamlit as st
import tempfile
from roboflow import Roboflow
from PIL import Image
import africastalking

rf = Roboflow(api_key="aQONY7aSjUN7H1sSqu0s")
project = rf.workspace().project("e-waste-detection-model")
model = project.version(3).model

# Page configuration
st.set_page_config(page_title="e vision", page_icon="🚮")

#########################################################
user_phone = st.text_input("Enter Your Phone Number: ")

# Display the entered text
if user_phone:
    st.write(f"You entered: {user_phone}")
    
class AIRTIME:
    def __init__(self):
		# Set your app credentials
        self.username = "mild.ke"
        self.api_key = "daae7b33a138c168292fbe863cd135d3b33b5f768639cef6f70d6ba141e0d8b1"

        # Initialize the SDK
        africastalking.initialize(self.username, self.api_key)

        # Get the airtime service
        self.airtime = africastalking.Airtime

    def send(self):
        # Set phone_number in international format
        phone_number = user_phone

        # Set The 3-Letter ISO currency code and the amount
        amount ="05.00"
        currency_code = "KES"

        try:
                # That's it hit send and we'll take care of the rest
                responses = self.airtime.send(phone_number=phone_number, amount=amount, currency_code=currency_code)
                print (responses)
        except Exception as e:
                print ("Encountered an error while sending airtime:%s" %str(e))

####################################
# Reward User
tokens = 0

st.title("e-vision")
st.markdown("Detect E-WASTE using AI")

uploaded_file = st.file_uploader(label="Upload Image", type=["jpg", "jpeg", "png"], key="1")
if uploaded_file is not None:
    # Save the uploaded image to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
        temp_file.write(uploaded_file.read())

    # Perform prediction on the saved image
    try:
        st.spinner("Processing.......")
        prediction = model.predict(temp_path, confidence=40, overlap=30)
        tokens += 5
        st.success(f"Tokens = {tokens}")
        if __name__ == '__main__':
    		AIRTIME().send()
        
    except:
        st.info("Failed, Upload another image")
            
    # Save the prediction image manually
    prediction_image_path = "prediction.jpg"
    prediction.save(prediction_image_path)

    # Display the prediction result using PIL
    prediction_image = Image.open(prediction_image_path)
    st.image(prediction_image, caption="Prediction Image", use_column_width=True)
