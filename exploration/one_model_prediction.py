import streamlit as st
from PIL import Image
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog
import numpy as np
import cv2

# Streamlit App
st.title("Bike Assessment Demo")
st.write("Upload an image to see object detection in action!")

# Category Mapping Dictionary
category_set_correct = {
    "FRAME_EXCELLENT": 4,
    "FRAME_GOOD": 3,
    "FRAME_POOR": 2,
    "FRAME_DEGRADED": 1,
}

# Configure Detectron2
@st.cache_resource
def load_model():
    cfg = get_cfg()
    cfg.merge_from_file("/workspaces/bike_demo/models/frame_custom_dataset.yaml")  # Update with your config file path
    cfg.MODEL.WEIGHTS = "/workspaces/bike_demo/models/frame_model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for predictions
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
    return DefaultPredictor(cfg)

predictor = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Run prediction
    st.write("Processing...")
    img_array = np.array(image)
    outputs = predictor(img_array)

    # Get predictions and the highest confidence score
    instances = outputs["instances"]
    scores = instances.scores.cpu().numpy()
    classes = instances.pred_classes.cpu().numpy()

    # Get the index of the highest confidence prediction
    best_prediction_idx = np.argmax(scores)
    best_score = scores[best_prediction_idx]
    best_class = classes[best_prediction_idx]

    # Map class to custom category
    # Assuming class index corresponds to one of the category_set_correct keys
    category_mapping = list(category_set_correct.keys())[best_class]
    
    # Display best prediction
    st.write(f"Best Prediction: {category_mapping} with Confidence: {best_score:.2f}")

    # Visualize predictions with the best one highlighted
    v = Visualizer(img_array[:, :, ::-1], scale=0.8)   #  MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
    v = v.draw_instance_predictions(instances[best_prediction_idx:best_prediction_idx+1].to("cpu"))

    # Display results
    st.image(v.get_image()[:, :, ::-1], caption="Predicted Image", use_container_width=True)
