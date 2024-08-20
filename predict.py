import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input # type: ignore
import numpy as np
import sys

# Mapping from class index to the fruit/vegetable name
CLASS_NAMES = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 
    'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 
    'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 
    'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 
    'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 
    'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def predict_image(model, image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = CLASS_NAMES[predicted_class_index]

    return predicted_class_name

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

# Get image path from command line argument
    image_path = sys.argv[1]  
    model_path = 'predictfruit.keras'
    
    model = load_model(model_path)
    result = predict_image(model, image_path)
    print(f'Predicted class: {result}')

if __name__ == "__main__":
    main()
