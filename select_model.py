import os
import streamlit as st
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
import numpy as np
from PIL import Image
import torch

# Streamlit App Title
st.title("Object Detection with Multiple Models")
st.write("Upload an image and choose a model to see object detection in action!")

# Updated Category Mapping Dictionary
category_set_correct = {
    "EXCELLENT": 4,
    "GOOD": 3,
    "POOR": 2,
    "DEGRADED": 1,
}

# Path to the folder containing models
MODEL_FOLDER = "/workspaces/bike_demo/models"  # Update to your folder path

# Load all models from the folder
@st.cache_resource
def load_models(folder_path):
    models = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".pth"):
            # Extract the object name from the filename
            object_name = filename.split("_model_final.pth")[0]
            config_file = os.path.join(folder_path, f"{object_name}_custom_dataset.yaml")
            if os.path.exists(config_file):
                # Load the model configuration and weights
                cfg = get_cfg()
                cfg.merge_from_file(config_file)
                cfg.MODEL.WEIGHTS = os.path.join(folder_path, filename)
                cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
                cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
                models[object_name] = DefaultPredictor(cfg)
    return models

models = load_models(MODEL_FOLDER)

# Dropdown to select a model
if models:
    selected_model = st.selectbox("Choose a model:", list(models.keys()))
else:
    st.error("No models found in the specified folder.")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file and selected_model:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Run prediction with the selected model
    st.write("Processing...")
    predictor = models[selected_model]
    img_array = np.array(image)
    outputs = predictor(img_array)

    # Get predictions and the highest confidence score
    instances = outputs["instances"]
    scores = instances.scores.cpu().numpy()
    classes = instances.pred_classes.cpu().numpy()

    if len(scores) > 0:
        # Get the index of the highest confidence prediction
        best_prediction_idx = np.argmax(scores)
        best_score = scores[best_prediction_idx]
        best_class = classes[best_prediction_idx]

        # Map class to custom category
        category_mapping = list(category_set_correct.keys())[best_class]

        # Display best prediction
        st.write(f"Best Prediction: {category_mapping} with Confidence: {best_score:.2f}")

        # Visualize predictions with the best one highlighted
        v = Visualizer(img_array[:, :, ::-1], scale=0.8)  # MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
        v = v.draw_instance_predictions(instances[best_prediction_idx:best_prediction_idx+1].to("cpu"))

        # Display results
        st.image(v.get_image()[:, :, ::-1], caption="Predicted Image", use_container_width=True)
    else:
        st.warning("No predictions made by the model.")
