# Faster R-CNN Implementation with PyTorch

This repository contains a project on implementing a **Faster R-CNN (FRCNN)** model for object detection using **PyTorch**, guided by the book *Modern Computer Vision with PyTorch (2nd Edition)*. The project is developed and executed in a Jupyter Notebook hosted on **Kaggle**.

---

## Features

- **Dataset**: The project uses the **Open Images Dataset** focusing on detecting buses and trucks.
- **Selective Search**: Utilized for generating region proposals.
- **Faster R-CNN Architecture**:
  - Backbone: Pre-trained VGG16.
  - Region of Interest (ROI) Pooling: Used for extracting features from proposed regions.
  - Classifier and Bounding Box Regressor: Implemented for object classification and localization.
- **Custom Dataset**: Created to handle Open Images format and integrate region proposals with ground truth.
- **Training and Validation**:
  - Loss Functions: Cross-Entropy Loss for classification and Smooth L1 Loss for bounding box regression.
  - Metrics: Includes accuracy and Intersection over Union (IoU).
- **Inference**: Demonstrates predictions on test images with visualization of detected objects and bounding boxes.

---

## Dataset

The **Open Images Dataset** subset is downloaded directly using the Kaggle API. The images and bounding box annotations focus on identifying buses and trucks.

### Structure:
- Images: Stored in `images/images/`.
- Annotations: Stored in `df.csv`.

---

## Requirements

To run the notebook, the following Python libraries are required:

- `torch`
- `torchvision`
- `torch_snippets`
- `selectivesearch`
- `cv2` (OpenCV)
- `numpy`
- `pandas`
- `matplotlib`
- `Pillow`

Install dependencies with:
```bash
pip install torch torchvision torch_snippets selectivesearch opencv-python pandas matplotlib pillow
```

---

## How to Run

1. **Clone the Notebook**: Open the notebook file on Kaggle or Colab from the provided URL.
2. **Download Dataset**: The dataset is downloaded programmatically within the notebook using the Kaggle API.
3. **Run the Notebook**: Execute the cells sequentially to:
   - Preprocess the dataset.
   - Train the Faster R-CNN model.
   - Test the model on unseen images.

---

## Training Details

- **Batch Size**: 2
- **Optimizer**: SGD with a learning rate of 0.001.
- **Epochs**: 5
- **Training Process**: 
  - The model is trained to classify regions as foreground (buses/trucks) or background.
  - Bounding box regression adjusts proposals for accurate localization.

---

## Results

- **Predictions**: The `test_predictions` function demonstrates the model's ability to detect objects in test images, showcasing bounding boxes and class labels.
- **Visualization**: Output includes side-by-side comparisons of original images and predictions.

---

## Acknowledgments

- Inspired and guided by the book *Modern Computer Vision with PyTorch (2nd Edition)*.
- Utilizes Kaggle's computational resources for training and evaluation.

---

## License

This project is open-source under the MIT License. Feel free to use and modify it for educational and research purposes.