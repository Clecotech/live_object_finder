from PIL import Image
import streamlit as st
from ultralytics import YOLO

def load_model():
    return YOLO("yolo11n.pt")

def query(image, model):
    results = model(image)
    result = results[0]
    result.save(filename="result.jpg")
    return "result.jpg", result

def main():
    st.title("Live Object Finder using YOLO11")
    
    model = load_model()
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Objects"):
            with st.spinner("Detecting objects..."):
                result_image, result = query(image, model)
                
            st.image(result_image, caption="Image with Bounding Boxes", use_column_width=True)
                
            st.write("Detected Objects with there possibility :")
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])*100
                st.markdown(f"- **{class_name}**: {confidence:.2f}%")
        else:
            st.write("Object detection completed!")


if __name__ == "__main__":
    main()