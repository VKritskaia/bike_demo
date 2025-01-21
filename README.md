# Bike Assessment Demo with Detectron2

This Streamlit app demonstrates object detection using multiple Detectron2 models. It processes an uploaded image, runs predictions using pre-trained models, and displays the best predictions with bounding boxes and confidence scores.

## Features
- Upload an image to get predictions for multiple object detection models.
- Automatically selects the best prediction from each model.
- Displays results with bounding boxes and confidence scores.
- Provides a summary table of predictions for easy review.

## How It Works
1. **Models**: Each model is a combination of `.pth` (weights) and `.yaml` (configuration) files stored in a specified directory.
2. **Prediction Workflow**:
   - All models process the uploaded image.
   - The highest-confidence prediction from each model is selected.
   - Bounding boxes and categories are displayed on the image.
3. **Category Mapping**: Predictions are mapped to human-readable categories:
   ```python
   category_set_correct = {
       "EXCELLENT": 4,
       "GOOD": 3,
       "POOR": 2,
       "DEGRADED": 1,
   }
4. **Output:**
    - Annotated image with predictions.
    - Summary table of all model predictions.

## Installation

**Prerequisites**
- Python 3.8+
- Detectron2 installed
- Streamlit installed

**Install Dependencies**
  ```bash
  pip install -r requirements.txt
  ```

## Usage
1. Clone the repository and navigate to the project directory:
  ```bash
  git clone <your-repo-url>
  cd <project-directory>
  ```
2. Start the Streamlit app:
  ```bash
  streamlit run app.py
  ```
4. Upload an image in the app to view predictions.

## Directory Structure
```plaintext
.
├── app.py                 # Streamlit app
├── exploration/
│   ├── one_model_prediction.py
│   └── select_model.py
├── models/                # Directory containing model files
│   ├── frame_model_final.pth
│   ├── frame_model_config.yaml
│   ├── other_model_final.pth
│   └── other_model_config.yaml
├── .gitignore
├── README.md
├── requirements.txt       # Python dependencies
└── streamlit_app.py       # Main app script
