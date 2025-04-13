# Artifact Detection in Generated Images

## Task Description

In this project, we train a binary image classification model to detect artifacts in generated images. Artifacts may include visual distortions such as extra fingers, hands, text, misplaced facial elements, or eyes not looking at the camera. The goal is to automate the quality assurance (QA) process for generated content.

This solution was developed as part of a Kaggle-like challenge and includes:
- Training and validation using convolutional neural networks (CNN)
- Micro F1-score evaluation
- Model saving for later inference

---

## Project Structure
    project_folder/
    ├── trainee_dataset/               # Folder with images (2 classes)
    │   ├── test/                         # Images with test artifacts
    │   └── traine/                         # Images with traine artifacts
    ├── artifact_detection_train.ipynb # Jupyter notebook with training code
    ├── requirements.txt               # File with dependencies
    ├── artifact_detection_model.keras # Trained model (saved after training)
    └── README.md

## Run the Code

1. Clone the repository
2. Create a virtual environment (optional)
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
3. Install dependencies (optional)
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Jupyter notebook
   ```bash
    jupyter notebook artifact_detection_train.ipynb
    ```