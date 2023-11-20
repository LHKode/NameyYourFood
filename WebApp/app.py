import streamlit as st
import PIL
from ultralytics import YOLO

st.title("NAME YOUR FOOD")
col1, col2 = st.columns(2)

model = YOLO('runs/detect/train41/weights/best.pt')

file = st.file_uploader('Upload image', type=["jpg", "jpeg", "png", "bmp", "webp"])

if not (file is None):
    with col1:
        uploaded_image = PIL.Image.open(file)
        st.image(file, caption="Uploaded Image", use_column_width=True)

        results = model.predict(uploaded_image)
        
    with col2:
        boxes = results[0].boxes
        res_plotted = results[0].plot()[:, :, ::-1]
        st.image(res_plotted, caption='Detected Image', use_column_width=True)

