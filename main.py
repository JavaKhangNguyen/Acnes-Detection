import streamlit as st
import torch
import torchvision
from PIL import Image
import torchvision.transforms as T
from torchvision import transforms
import os
from ultralytics import YOLO

#Styling
style = """
<style>
    #MainMenu{visibility: hidden;}
</style>
"""

st.set_page_config(page_title='Acnes detection')
st.markdown(style, unsafe_allow_html=True)

model = YOLO('./Model/acnes(8m).pt')

image_ext = ["png", "jpg", "jpeg", "heic", "heif"]


with open("./test/test.jpg", "rb") as file:
    btn = st.download_button(
            label="Download image for testing",
            data=file,
            file_name="tes.jpg",
            mime="image/jpg"
          )

uploaded_file = st.file_uploader("Or choose a file")
if uploaded_file is not None:
    ext_position = len(uploaded_file.name.split('.')) - 1
    file_ext = uploaded_file.name.split('.')[ext_position]
    if file_ext in image_ext:
        image = Image.open(uploaded_file)
        st.header("Uploaded Image")
        st.image(image)
        results = model.predict(image)
        result = results[0]
        st.write(f'Result: {len(results[0].boxes)} acnes detected')
        st.header("Predictions Result")
        result_plotted = result.plot()
        st.image(result_plotted,
                    caption='Detected result',
                    channels="BGR",
                    use_column_width=True,
                                        )
        for box in result.boxes:
            class_id = result.names[box.cls[0].item()]
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            conf = (round(box.conf[0].item(), 2))  
            percentage_conf = f"{conf * 100:.0f}%"
            st.text(f"Object type: {class_id}")
            st.text(f"Coordinates: {cords}")
            st.text(f"Confidence: {percentage_conf}")
            st.divider()
st.image('./logo/AIoT.png', width=700)
