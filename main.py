from ultralytics import YOLO
import streamlit as st
from PIL import Image


def load_model():
    # Load the YOLOv8 model
    model = YOLO("yolo11n.pt")  # Load a pretrained YOLOv8 model
    return model


def detect_objects(model, image):
    opened_image = Image.open(image)

    results = model(opened_image)
    return results[0].plot()


def main():
    st.set_page_config(
        page_title="YOLOv8 Object Detection", page_icon="🤖", layout="centered"
    )

    st.title("YOLOv8 Object Detection")
    st.write(
        "Upload an image to detect objects using the YOLOv8 model. The model is trained on the COCO dataset."
    )

    @st.cache_resource
    def cache_model():
        return load_model()

    model = cache_model()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        btn = st.button("Classify Image")

        if btn:
            with st.spinner("Detecting objects..."):
                results = detect_objects(model, uploaded_file)
                st.image(results, caption="Detected Image", use_container_width=True)
                st.success("Detection complete!")


if __name__ == "__main__":
    main()
