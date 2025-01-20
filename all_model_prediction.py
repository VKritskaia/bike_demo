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
st.write("Upload an image to see predictions from all models!")

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
                cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
                cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
                models[object_name] = DefaultPredictor(cfg)

    return models

models = load_models(MODEL_FOLDER)

# if not models:
#     st.error("No models found in the specified folder.")
#     st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Convert image to numpy array
    img_array = np.array(image)
    st.write("Processing...")

    # Dictionary to store the best prediction for each model
    predictions = []

    for model_name, predictor in models.items():
        # Run prediction
        outputs = predictor(img_array)

        # Get predictions and check for empty instances
        instances = outputs["instances"]
        if len(instances) == 0:
            st.warning(f"No predictions made by the model: {model_name}")
            continue

        scores = instances.scores.cpu().numpy()
        classes = instances.pred_classes.cpu().numpy()

        if len(scores) > 0:
            # Get the index of the highest confidence prediction
            best_prediction_idx = np.argmax(scores)
            best_score = scores[best_prediction_idx]
            best_class = classes[best_prediction_idx]
            
            # Map class to custom category
            category_mapping = list(category_set_correct.keys())[best_class]

            st.write(model_name, instances.pred_boxes)
            # Add the best prediction details to the list
            predictions.append({
                "model_name": model_name,
                "category": category_mapping,
                "confidence": best_score,
                "bbox": instances.pred_boxes[best_prediction_idx].tensor.cpu().numpy()[0]
            })
    
    if predictions:
        # Visualize predictions on the image
        visualizer = Visualizer(img_array[:, :, ::-1], scale=0.8)
        for prediction in predictions:
            bbox = prediction["bbox"]
            category = prediction["category"]
            confidence = prediction["confidence"]
            label = f"{category}: {confidence:.2f}"
            visualizer.draw_box(bbox, edge_color="green")
            visualizer.draw_text(label, bbox[:2], font_size=12, color="white", box_facecolor="green")

        # Display the results
        st.image(visualizer.get_image()[:, :, ::-1], caption="Predictions from All Models", use_container_width=True)
    else:
        st.warning("No predictions made by any model.")

    # Display prediction details in a table
    st.write("Prediction Summary:")
    for prediction in predictions:
        st.write(
            f"Model: {prediction['model_name']}, "
            f"Category: {prediction['category']}, "
            f"Confidence: {prediction['confidence']:.2f}"
        )