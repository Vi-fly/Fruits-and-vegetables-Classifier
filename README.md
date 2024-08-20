# Fruits and Vegetables Classifier

This project is a deep learning-based image classifier built using TensorFlow and Keras. The model is designed to classify images into one of 36 different categories of fruits and vegetables. The classifier is based on the MobileNetV3Small architecture, which is a lightweight model suitable for mobile and embedded applications.

## Project Structure

- **Training Script**: `train_model.py` - Contains the code to train the model.
- **Model File**: `predictfruit.keras` - The saved Keras model.

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
- 
## Features

- **36-Class Classification:** The model can classify images of 36 different fruits and vegetables.
- **Preprocessing:** Images are preprocessed using the MobileNetV3 preprocess input function.
- **Transfer Learning:** The model leverages a pre-trained MobileNetV3Small network, with additional fully connected layers for classification.
- **Early Stopping:** The training process includes early stopping to prevent overfitting.
- **Easy Deployment:** The model is saved in a `.keras` file format, making it easy to load and use for predictions.

## Model Performance

The model was trained on a diverse dataset of fruits and vegetables and achieved the following performance metrics:

- **Accuracy:** 97%
- **Precision, Recall, F1-Score:** Detailed in the classification report below.

### Classification Report

| Class Name       | Precision | Recall | F1-Score | Support |
|------------------|-----------|--------|----------|---------|
| Apple            | 0.73      | 0.80   | 0.76     | 10      |
| Banana           | 1.00      | 0.78   | 0.88     | 9       |
| Beetroot         | 1.00      | 1.00   | 1.00     | 10      |
| Bell Pepper      | 0.90      | 1.00   | 0.95     | 9       |
| Cabbage          | 1.00      | 1.00   | 1.00     | 10      |
| ...              | ...       | ...    | ...      | ...     |
| Watermelon       | 1.00      | 1.00   | 1.00     | 10      |
| **Total**        | **0.97**  | **0.97** | **0.97** | **340** |

## Installation

To clone and use this model, follow these steps:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Vi-fly/Fruits-and-vegetables-Classifier.git
   cd Fruits-and-vegetables-Classifier
   ```

2. **Install the Required Packages**

   Install the necessary Python packages using the provided `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Predicting with the Model

Once you have the model loaded, you can make predictions on any image using the following script.

```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
import sys

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def predict_image(model, image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    return predicted_class

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = 'test.png'  # Replace with your image path
    model_path = 'predictfruit.keras'
    
    model = load_model(model_path)
    result = predict_image(model, image_path)
    print(f'Predicted class index: {result}')

    sys.exit(0)

if __name__ == "__main__":
    main()
```

### 2. Running Predictions

You can run predictions on any image using the following command:

```bash
python predict.py path/to/your/image.jpg
```

The script will output the predicted class for the given image.

## Requirements

The following packages are required to run the model and make predictions:

```plaintext
numpy
matplotlib
seaborn
tensorflow
pandas
```

Install these dependencies with:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The model architecture is based on MobileNetV3Small.
- The dataset includes a variety of fruits and vegetables from multiple sources.

---

This README should provide a clear and comprehensive guide for users who wish to clone the repository, set up the environment, and use the model for predictions.
