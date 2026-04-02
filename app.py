import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# 1. THE BRAIN: Deep CNN (High Efficiency)
def get_model():
    try:
        # Tries to load the saved model
        return tf.keras.models.load_model('robust_cnn_model.h5')
    except:
        # Trains a professional-grade CNN if no model exists
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
        
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
            tf.keras.layers.BatchNormalization(), # Stabilizes for thin/thick lines
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.4), # Makes model flexible for messy styles
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=5) 
        model.save('robust_cnn_model.h5')
        return model

model = get_model()

# 2. THE EYE: Universal Pre-processing
def predict(data):
    if data is None or 'composite' not in data:
        return "Please draw a number!"
    
    # Get the "ink" layer (Alpha channel)
    img = Image.fromarray(data['composite'][:,:,3].astype('uint8'))
    
    # DYNAMIC CROP: This is the "Efficiency" secret. 
    # It finds the digit even if it's tiny or in a corner.
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    
    # SCALE & CENTER: Centers the digit on a 28x28 black background
    width, height = img.size
    max_dim = max(width, height)
    new_img = Image.new('L', (max_dim + 20, max_dim + 20), 0)
    new_img.paste(img, ((max_dim - width) // 2 + 10, (max_dim - height) // 2 + 10))
    
    # FINAL FORMATTING
    img = new_img.resize((28, 28))
    img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0
    
    # PREDICT
    preds = model.predict(img_array)[0]
    return {str(i): float(preds[i]) for i in range(10)}

# 3. INTERFACE
interface = gr.Interface(
    fn=predict, 
    inputs=gr.Sketchpad(label="Draw Naturally (Any Size/Style)", type="numpy"), 
    outputs=gr.Label(num_top_classes=3),
    title="Universal Digit Recognizer (CNN)",
    description="Optimized for Task 5. This system handles various handwriting styles, sizes, and positions."
)

interface.launch()
