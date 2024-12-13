import os
import streamlit as st
from paddleocr import PaddleOCR
from PIL import Image
import tempfile
import json

ocr = PaddleOCR(use_angle_cls=True, lang='en')

st.title("Image Text Extraction")

# Check if this is a Postman API request
if st.experimental_get_query_params().get("api", ["false"])[0].lower() == "true":
    from io import BytesIO

    # Get file input from Postman (via JSON body)
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png', 'webp'])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image:
            image = Image.open(uploaded_file)
            image.save(temp_image.name)

            # Perform OCR
            result = ocr.ocr(temp_image.name)
            detected_text = "\n".join([line[1][0] for line in result[0]])

            # Return JSON response for Postman
            st.json({"extracted_text": detected_text})
            os.remove(temp_image.name)

else:
    # For the normal Streamlit app
    uploaded_files = st.file_uploader(
        "Choose image files",
        type=['jpg', 'jpeg', 'png', 'webp'],
        accept_multiple_files=True
    )

    if uploaded_files:
        detected_text = ""

        for idx, uploaded_file in enumerate(uploaded_files, start=1):
            # Display the uploaded image
            image = Image.open(uploaded_file)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image:
                image.save(temp_image.name)
                result = ocr.ocr(temp_image.name)
                page_text = "\n".join([line[1][0] for line in result[0]])
                detected_text += page_text.strip() + "\n\n"
                os.remove(temp_image.name)

        st.text_area("Extracted Text", detected_text)


