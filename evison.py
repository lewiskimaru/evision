import streamlit as st
import tempfile
from roboflow import Roboflow
from PIL import Image

rf = Roboflow(api_key="aQONY7aSjUN7H1sSqu0s")
project = rf.workspace().project("e-waste-detection-model")
model = project.version(3).model

st.title("Image")
st.markdown("Upload an Image and have it decoded")

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
        st.success("Results")
    except:
        st.info("Failed, Upload another image")
            
    # Save the prediction image manually
    prediction_image_path = "prediction.jpg"
    prediction.save(prediction_image_path)

    # Display the prediction result using PIL
    prediction_image = Image.open(prediction_image_path)
    st.image(prediction_image, caption="Prediction Image", use_column_width=True)
