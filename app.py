import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# 1. BRAIN: Load or Train the Model
def get_model():
    try:
        # Tries to load the saved model if it exists in your Space
        return tf.keras.models.load_model('digit_model.h5')
    except:
        # If not, it trains a high-accuracy CNN on the spot
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
        
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.Dropout(0.3), 
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=3) 
        model.save('digit_model.h5')
        return model

model = get_model()

# 2. ENGINE: The logic that makes it "Robust"
def predict(data):
    if data is None or 'composite' not in data:
        return "Please draw a digit"
    
    # Get the drawing layer (Alpha channel)
    raw_image = Image.fromarray(data['composite'][:,:,3].astype('uint8'))
    
    # AUTO-CROP: Find the number and cut it out (Handles small/corner drawings)
    bbox = raw_image.getbbox()
    if bbox:
        raw_image = raw_image.crop(bbox)
    
    # ADD PADDING: Centers the digit to look like the MNIST dataset
    raw_image = ImageOps.expand(raw_image, border=20, fill=0)
    
    # RE-SIZE: Force into 28x28 pixels
    raw_image = raw_image.resize((28, 28)).convert('L')
    
    # NORMALIZE: Convert to array and scale
    final_img = np.array(raw_image) / 255.0
    final_img = final_img.reshape(1, 28, 28, 1)
    
    # PREDICT
    preds = model.predict(final_img)[0]
    return {str(i): float(preds[i]) for i in range(10)}

# 3. INTERFACE: The Web App look
interface = gr.Interface(
    fn=predict, 
    inputs=gr.Sketchpad(label="Draw your digit here", type="numpy"), 
    outputs=gr.Label(num_top_classes=3),
    title="Handwritten Digit Recognition System",
    description="Draw a digit (0-9). This app uses a CNN with dynamic cropping to ensure high accuracy for all styles."
)

interface.launch()
