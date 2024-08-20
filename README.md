# Fruit and Vegetable Image Classification Model

## Overview

This project demonstrates an image classification model trained to recognize and classify 36 different fruits and vegetables. Using TensorFlow and MobileNetV3, the model achieves high accuracy in identifying various produce types. This document provides an overview of the model, its performance, and how to use it.

## Project Structure

- **Training Script**: `train_model.py` - Contains the code to train the model.
- **Model File**: `predictfruit.keras` - The saved Keras model.
- **TFLite Model File**: `predictfruit.tflite` - The converted TensorFlow Lite model for mobile deployment.

## Model Details

### Architecture

- **Base Model**: MobileNetV3Small, pre-trained on ImageNet, used as the feature extractor.
- **Custom Layers**:
  - **Dense Layers**: Two dense layers with 128 units each and ReLU activation.
  - **Output Layer**: A dense layer with 36 units (one for each class) and softmax activation for classification.

### Training

- **Data Augmentation**: Includes rotation, zoom, shift, shear, and horizontal flip.
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy
- **Early Stopping**: Monitors validation loss with a patience of 2 epochs.

### Model Performance

The model was evaluated using various metrics. Below is a detailed classification report showing precision, recall, f1-score, and support for each class.

#### Classification Report

| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| apple         | 0.73      | 0.80   | 0.76     | 10      |
| banana        | 1.00      | 0.78   | 0.88     | 9       |
| beetroot      | 1.00      | 1.00   | 1.00     | 10      |
| bell pepper   | 0.90      | 1.00   | 0.95     | 9       |
| cabbage       | 1.00      | 1.00   | 1.00     | 10      |
| capsicum      | 1.00      | 0.90   | 0.95     | 10      |
| carrot        | 1.00      | 1.00   | 1.00     | 8       |
| cauliflower   | 1.00      | 1.00   | 1.00     | 10      |
| chilli pepper | 0.88      | 1.00   | 0.93     | 7       |
| corn          | 0.83      | 1.00   | 0.91     | 10      |
| cucumber      | 1.00      | 1.00   | 1.00     | 10      |
| eggplant      | 1.00      | 1.00   | 1.00     | 10      |
| garlic        | 1.00      | 1.00   | 1.00     | 10      |
| ginger        | 1.00      | 1.00   | 1.00     | 10      |
| grapes        | 1.00      | 1.00   | 1.00     | 9       |
| jalepeno      | 1.00      | 1.00   | 1.00     | 9       |
| kiwi          | 1.00      | 1.00   | 1.00     | 10      |
| lemon         | 1.00      | 1.00   | 1.00     | 7       |
| lettuce       | 1.00      | 1.00   | 1.00     | 9       |
| mango         | 1.00      | 1.00   | 1.00     | 10      |
| onion         | 1.00      | 1.00   | 1.00     | 9       |
| orange        | 0.88      | 1.00   | 0.93     | 7       |
| paprika       | 0.90      | 0.90   | 0.90     | 10      |
| pear          | 0.91      | 1.00   | 0.95     | 10      |
| peas          | 1.00      | 1.00   | 1.00     | 9       |
| pineapple     | 1.00      | 1.00   | 1.00     | 10      |
| pomegranate   | 1.00      | 1.00   | 1.00     | 10      |
| potato        | 1.00      | 0.80   | 0.89     | 10      |
| raddish       | 1.00      | 1.00   | 1.00     | 8       |
| soy beans     | 1.00      | 1.00   | 1.00     | 10      |
| spinach       | 1.00      | 1.00   | 1.00     | 10      |
| sweetcorn     | 1.00      | 0.80   | 0.89     | 10      |
| sweetpotato   | 1.00      | 0.90   | 0.95     | 10      |
| tomato        | 1.00      | 1.00   | 1.00     | 10      |
| turnip        | 0.91      | 1.00   | 0.95     | 10      |
| watermelon    | 1.00      | 1.00   | 1.00     | 10      |

- **Accuracy**: 0.97
- **Macro Average**: Precision: 0.97, Recall: 0.97, F1-Score: 0.97
- **Weighted Average**: Precision: 0.97, Recall: 0.97, F1-Score: 0.97

## Model Files

1. **`predictfruit.keras`**: The saved Keras model file.
2. **`predictfruit.tflite`**: The TensorFlow Lite model file for mobile deployment.

## Usage

```markdown
# Fruits and Vegetables Classifier

This repository contains a pre-trained model for classifying images of fruits and vegetables. 

## How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Fruits-and-vegetables-Classifier.git
cd Fruits-and-vegetables-Classifier
```

### 2. Set Up the Environment

Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
```

Activate the virtual environment:

- **Windows**:

  ```bash
  venv\Scripts\activate
  ```

- **macOS/Linux**:

  ```bash
  source venv/bin/activate
  ```

Install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Make Predictions

To make predictions, use the `predict.py` script. Provide the path to the image as a command-line argument:

```bash
python predict.py path_to_your_image.jpg
```

Replace `path_to_your_image.jpg` with the path to the image you want to classify.

The script will output the predicted class index.

```

### Model Inference

To use the model for inference, you need to preprocess the input images to match the model's input format. Ensure the image is resized to 224x224 pixels and normalized before feeding it into the model.

## Requirements

- **Python**: 3.7 or later
- **TensorFlow**: 2.x
- **NumPy**: 1.19 or later
- **Pandas**: 1.1 or later
- **Matplotlib**: 3.3 or later

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
