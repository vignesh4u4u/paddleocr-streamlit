import os
import streamlit as st
from paddleocr import PaddleOCR
from PIL import Image

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

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
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded Image {idx}", use_column_width=True)

        # Save the image temporarily for processing
        temp_image_path = f"temp_image_{idx}.png"
        image.save(temp_image_path)

        # Perform OCR on the image
        result = ocr.ocr(temp_image_path)
        page_text = ""
        for line in result[0]:  # result[0] contains OCR text lines
            page_text += line[1][0].strip() + " "

        # Append the text from the current image
        detected_text += page_text.strip() + "\n\n"
        st.write(f"Extracted Text from Image {idx}:")
        st.write(page_text.strip())

        # Remove the temporary image
        os.remove(temp_image_path)

    # Display all the detected text from the images
    st.subheader("Combined Extracted Text:")
    st.text_area("Detected Text", detected_text, height=200)
