import numpy as np
import tensorflow as tf
# Path to your image and model
test_image_path = 'Tomato_je.jpg'
input_size = (150, 150)
classes = [
    'apple',
    'banana',
    'beetroot',
    'bell pepper',
    'cabbage',
    'capsicum',
    'carrot',
    'cauliflower',
    'chilli pepper',
    'corn',
    'cucumber',
    'eggplant',
    'garlic',
    'ginger',
    'grapes',
    'jalepeno',
    'kiwi',
    'lemon',
    'lettuce',
    'mango',
    'onion',
    'orange',
    'paprika',
    'pear',
    'peas',
    'pineapple',
    'pomegranate',
    'potato',
    'raddish',
    'soy beans',
    'spinach',
    'sweetcorn',
    'sweetpotato',
    'tomato',
    'turnip',
    'watermelon'
]
  # Replace with your actual class names

# Load model
loaded_model = tf.keras.models.load_model('model.keras')

# Load and preprocess image
img = tf.keras.preprocessing.image.load_img(test_image_path, target_size=input_size)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Apply normalization if your model was trained with it
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True
)

img_generator = test_datagen.flow(img_array, batch_size=1)

# Predict
prediction = loaded_model.predict(next(img_generator))
prediction_label = np.argmax(prediction)

print("Prediction:", classes[prediction_label])
