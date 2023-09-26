import streamlit as st
import torch
import torchvision
import os
import torchvision.transforms as T

from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

#Styling
style = """
<style>
    #MainMenu{visibility: hidden;}
</style>
"""

st.set_page_config(page_title='Acnes detection')
st.markdown(style, unsafe_allow_html=True)

model = YOLO('./Model/acnes_v3(8m).pt')

image_ext = ["png", "jpg", "jpeg", "heic", "heif"]

st.text("Types of acnes that this model can classify: acne_scars, blackhead, cystic, ")
st.text("flat_wart, folliculitis, keloid, milium, papular, purulent, sebo-crystan-conglo,")
st.text("syringoma, whitehead")

with open("./test/test.jpg", "rb") as file:
    btn = st.download_button(
            label="Download image for testing",
            data=file,
            file_name="test.jpg",
            mime="image/jpg"
          )

uploaded_file = st.file_uploader("Or choose a file", accept_multiple_files=False, type=["png", "jpg", "jpeg", "heic", "heif"])
if uploaded_file is not None:
    ext_position = len(uploaded_file.name.split('.')) - 1
    file_ext = uploaded_file.name.split('.')[ext_position]
    if file_ext in image_ext:
        image = Image.open(uploaded_file)
        # Resize the image while maintaining aspect ratio
        max_size = 800
        img_width, img_height = image.size
        if img_width > max_size or img_height > max_size:
            # Calculate the aspect ratio
            aspect_ratio = img_width / img_height
            if img_width > img_height:
                new_width = max_size
                new_height = int(max_size / aspect_ratio)
            else:
                new_height = max_size
                new_width = int(max_size * aspect_ratio)
            image = image.resize((new_width, new_height), Image.ANTIALIAS)
            
        else:
            # If both width and height are smaller than max_size, upscale to max_size
            aspect_ratio = img_width / img_height
            if img_width > img_height:
                new_width = max_size
                new_height = int(max_size / aspect_ratio)
            else:
                new_height = max_size
                new_width = int(max_size * aspect_ratio)
            image = image.resize((new_width, new_height), Image.ANTIALIAS)
            
        st.header("Uploaded Image")
        st.image(image)
        with st.spinner("AI is processing your image"):
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
