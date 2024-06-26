import streamlit as st

from PIL import Image
from ultralytics import YOLO, YOLOv10

#Styling
style = """
<style>
    #MainMenu{visibility: hidden;}
</style>
"""

st.set_page_config(page_title='Acnes Detection')
st.markdown(style, unsafe_allow_html=True)

# model = YOLO('./Model/ACNE8M.pt')
model = YOLOv10('./Model/ACNE10.pt')

image_ext = ["png", "jpg", "jpeg", "heic", "heif"]


st.header("Acnes Detection")
st.markdown(
        """
        **Types of acnes that this model can classify: Acne scars, Blackhead, Cystic, Flat wart, Folliculitis, Keloid, Milium, Papular, Purulent, 
        Sebo-crystan-conglo, Syringoma, Whitehead**
        """
)
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
        #Resize image while maintaining aspect ratio
        size = 800, 800
        image.thumbnail(size, Image.Resampling.LANCZOS)
            
        st.header("Uploaded Image")
        st.image(image)
        with st.spinner("AI is processing your image"):
            results = model.predict(image, conf=0.5)
            
        
        result = results[0]
        st.success("✅ AI has finished the job! ✅")
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
            st.write(f"Object type: {class_id}")
            st.write(f"Coordinates: {cords}")
            st.write(f"Confidence: {percentage_conf}")
            st.divider()
    st.info("**TIP**: If the model cannot predict the uploaded image, try to preprocess it before uploading again "
            "The best result will be achieved if the image focus mainly on the acnes. ")

st.image('./logo/AIoT.png', width=700)
