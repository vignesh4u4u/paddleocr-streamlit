import os
import tempfile
import json
import streamlit as st
from paddleocr import PaddleOCR
from PIL import Image
from io import BytesIO

ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Check if the request is coming from Postman
if st.experimental_get_query_params().get("api", ["false"])[0].lower() == "true":
    st.title("API for OCR Text Extraction")

    # Receive the file from Postman
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png', 'webp'])

    if uploaded_file:
        detected_text = ""
        
        # Process the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image:
            image = Image.open(uploaded_file)
            image.save(temp_image.name)

            # Perform OCR
            result = ocr.ocr(temp_image.name)
            page_text = "\n".join([line[1][0] for line in result[0]])
            detected_text += page_text.strip() + "\n\n"

            # Cleanup temp file
            os.remove(temp_image.name)

        # Return the result in JSON format
        st.json({"extracted_text": detected_text.strip()})

else:
    # Regular Streamlit app for UI
    st.title("Image Text Extraction")

    # File uploader for multiple image files
    uploaded_files = st.file_uploader(
        "Choose image files",
        type=['jpg', 'jpeg', 'png', 'webp'],
        accept_multiple_files=True
    )

    if uploaded_files:
        detected_text = ""

        for idx, uploaded_file in enumerate(uploaded_files, start=1):
            # Process each uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image:
                image = Image.open(uploaded_file)
                image.save(temp_image.name)

                # Perform OCR
                result = ocr.ocr(temp_image.name)
                page_text = "\n".join([line[1][0] for line in result[0]])
                detected_text += page_text.strip() + "\n\n"

                # Cleanup temp file
                os.remove(temp_image.name)

        # Display extracted text
        st.write(detected_text.strip())



