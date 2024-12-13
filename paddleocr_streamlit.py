import os
import streamlit as st
from paddleocr import PaddleOCR
from PIL import Image

ocr = PaddleOCR(use_angle_cls=True, lang='en')

st.title("Image Text Extraction")

# File uploader to allow multiple file uploads
uploaded_files = st.file_uploader(
    "Choose image files", 
    type=['jpg', 'jpeg', 'png', 'webp'], 
    accept_multiple_files=True
)

# Check if files are uploaded
if uploaded_files:
    # Iterate over each uploaded file
    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        # Open the image
        image = Image.open(uploaded_file)
        
        # Run OCR on the image
        result = ocr.ocr(image)
        
        # Collect the extracted text for the current image
        page_text = ""
        for line in result[0]:
            page_text += line[1][0].strip() + " "  # Append the recognized text
        
        # Display the extracted text for each image
        st.write(f"Extracted Text from Image {idx}:")
        st.text(page_text.strip())  # Display the extracted text for the image
else:
    st.warning("Please upload some image files.")

    
