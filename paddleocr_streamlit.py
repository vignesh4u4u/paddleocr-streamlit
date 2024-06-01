import streamlit as st
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

st.title('Image Text Extraction')

uploaded_files = st.file_uploader("Choose image files", type=['jpg', 'jpeg', 'png', 'webp'], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        image_data = uploaded_file.read()
        result = ocr.ocr(image_data)
        text = ""
        for line in result:
            for word in line:
                text += word[1][0] + " "
        st.write(f"Extracted Text: {text}")



