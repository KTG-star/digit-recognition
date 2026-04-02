import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# 1. THE BRAIN: Deep CNN with Batch Normalization
def get_model():
    try:
        return tf.keras.models.load_model('universal_model.h5')
    except:
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
        
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
            tf.keras.layers.BatchNormalization(), # Makes the model ignore stroke thickness
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.4), # Highly flexible to messy handwriting
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # Train with more epochs for higher "perfection"
        model.fit(x_train, y_train, epochs=10) 
        model.save('universal_model.h5')
        return model

model = get_model()

# 2. THE EYE: Maximum Efficiency Pre-processing
def predict(data):
    if data is None: return None
    
    # Extract image and convert to Grayscale
    # Gradio 'composite' contains the drawing
    img = Image.fromarray(data['composite'].astype('uint8')).convert('L')
    
    # STEP A: THE INVERSION FIX
    # If the user draws black-on-white, we MUST flip it to white-on-black for the AI
    stat = np.array(img)
    if stat.mean() > 127:
        img = ImageOps.invert(img)
    
    # STEP B: DYNAMIC CROPPING (The "Any Position" Fix)
    # This finds the exact box where the ink is and ignores the rest of the screen
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    
    # STEP C: ASPECT RATIO PRESERVATION & CENTERING
    # This prevents the number from getting "squashed" when resizing
    width, height = img.size
    max_dim = max(width, height)
    # Create a square canvas with a margin
    container = Image.new('L', (max_dim + 40, max_dim + 40), 0)
    container.paste(img, ((max_dim - width) // 2 + 20, (max_dim - height) // 2 + 20))
    
    # STEP D: FINAL PREP
    img = container.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0
    
    # PREDICT
    preds = model.predict(img_array)[0]
    return {str(i): float(preds[i]) for i in range(10)}

# 3. INTERFACE
interface = gr.Interface(
    fn=predict, 
    inputs=gr.Sketchpad(label="Draw Any Number (Any Style/Size/Place)", type="numpy"), 
    outputs=gr.Label(num_top_classes=3),
    title="Universal Digit Recognition System",
    description="Optimized for Task 5: Uses CNN architecture with dynamic cropping and inversion handling."
)

interface.launch()
