import os
import streamlit as st
from paddleocr import PaddleOCR
from PIL import Image

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
        #st.image(image, caption=f"Uploaded Image {idx}", use_column_width=True)        
        temp_image_path = f"temp_image_{idx}.png"
        image.save(temp_image_path)        
        result = ocr.ocr(temp_image_path)
        page_text = ""
        for line in result[0]: 
            page_text += line[1][0].strip() + " "        
        detected_text += page_text.strip() + "\n\n"
        #st.write(f"Extracted Text from Image {idx}:")
        #st.write(page_text.strip())        
        os.remove(temp_image_path)

    
    st.write(detected_text)
