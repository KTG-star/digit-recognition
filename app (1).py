# ================================================================
#   CSC 309 — Task 5: Handwritten Digit Recognition
#   Web Application built with Streamlit + TensorFlow/Keras
#   Dataset: MNIST | Model: Convolutional Neural Network (CNN)
# ================================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import io
import time

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Digit Recognition — CSC 309",
    page_icon="🔢",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  .main-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #1a1410;
    margin-bottom: 0;
  }
  .sub-title {
    font-size: 0.85rem;
    color: #8a7f72;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
    margin-bottom: 28px;
  }
  .metric-card {
    background: #fdfaf4;
    border: 2px solid #1a1410;
    border-radius: 8px;
    padding: 18px 20px;
    box-shadow: 4px 4px 0 #1a1410;
    text-align: center;
  }
  .metric-num {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #c0392b;
  }
  .metric-label {
    font-size: 0.75rem;
    color: #8a7f72;
    text-transform: uppercase;
    letter-spacing: 1.5px;
  }
  .pred-box {
    background: #fdfaf4;
    border: 3px solid #1a1410;
    border-radius: 10px;
    padding: 28px;
    box-shadow: 6px 6px 0 #c0392b;
    text-align: center;
  }
  .pred-number {
    font-family: 'Syne', sans-serif;
    font-size: 6rem;
    font-weight: 800;
    color: #c0392b;
    line-height: 1;
  }
  .pred-conf {
    font-size: 1.6rem;
    font-weight: 600;
    color: #1a1410;
    margin-top: 8px;
  }
  .concept-box {
    background: #fdfaf4;
    border: 2px solid #e8e2d6;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 10px;
  }
  .concept-box h4 { color: #c0392b; margin-bottom: 4px; font-size: 0.9rem; }
  .concept-box p  { color: #3a3530; font-size: 0.85rem; line-height: 1.5; margin: 0; }
  .stProgress > div > div { background-color: #c0392b !important; }

  div[data-testid="stHorizontalBlock"] { align-items: start; }
</style>
""", unsafe_allow_html=True)


# ── Load & cache model ────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Train CNN on MNIST and cache it so it only trains once."""

    # Load MNIST
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Preprocess
    X_train = X_train.astype("float32") / 255.0
    X_test  = X_test.astype("float32")  / 255.0
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test  = X_test.reshape(-1, 28, 28, 1)

    # Build CNN
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(10,  activation='softmax'),
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train
    model.fit(X_train, y_train, epochs=5, batch_size=128,
              validation_split=0.1, verbose=0)

    # Evaluate
    _, acc = model.evaluate(X_test, y_test, verbose=0)

    return model, acc, X_test, y_test


# ── Header ───────────────────────────────────────────────────
st.markdown('<div class="main-title">🔢 Handwritten Digit Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">CSC 309 · Task 5 · Convolutional Neural Network · MNIST Dataset</div>', unsafe_allow_html=True)

# ── Load model with progress bar ─────────────────────────────
with st.spinner("⏳ Training CNN on MNIST dataset… (first load only, ~30 seconds)"):
    model, test_accuracy, X_test, y_test = load_model()

st.success(f"✅ Model ready! Test accuracy on 10,000 MNIST images: **{test_accuracy*100:.2f}%**")

# ── Metrics row ───────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-num">{test_accuracy*100:.1f}%</div><div class="metric-label">Test Accuracy</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><div class="metric-num">70K</div><div class="metric-label">MNIST Images</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><div class="metric-num">CNN</div><div class="metric-label">Model Type</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card"><div class="metric-num">10</div><div class="metric-label">Digit Classes</div></div>', unsafe_allow_html=True)

st.markdown("---")

# ── Main section: Draw + Predict ─────────────────────────────
st.subheader("✏️ Draw a Digit & Get a Prediction")
st.caption("Draw any digit from 0–9 in the box below, then click **Predict**.")

col_draw, col_result = st.columns([1, 1], gap="large")

with col_draw:
    st.markdown("**Drawing Canvas**")

    canvas_result = st_canvas(
        fill_color   = "rgba(0,0,0,0)",
        stroke_width = 18,
        stroke_color = "#FFFFFF",
        background_color = "#000000",
        height = 280,
        width  = 280,
        drawing_mode = "freedraw",
        key = "canvas",
    )

    predict_btn = st.button("🔍 Predict Digit", type="primary", use_container_width=True)
    clear_info  = st.caption("Tip: Draw thick and centred for best results.")

with col_result:
    st.markdown("**Prediction Result**")

    if predict_btn and canvas_result.image_data is not None:
        img_array = canvas_result.image_data

        # Check if canvas has any drawing
        if img_array[:, :, :3].sum() < 1000:
            st.warning("Canvas looks empty — please draw a digit first!")
        else:
            # Preprocess: resize to 28×28, grayscale, normalise
            img = Image.fromarray(img_array.astype("uint8"))
            img = img.convert("L")
            img = img.resize((28, 28), Image.LANCZOS)
            img_np = np.array(img).astype("float32") / 255.0
            img_np = img_np.reshape(1, 28, 28, 1)

            # Predict
            probs      = model.predict(img_np, verbose=0)[0]
            pred_digit = int(np.argmax(probs))
            confidence = float(probs[pred_digit]) * 100

            # Show result
            st.markdown(f"""
            <div class="pred-box">
              <div class="pred-number">{pred_digit}</div>
              <div class="pred-conf">{confidence:.1f}% confident</div>
              <div style="color:#8a7f72;font-size:0.8rem;margin-top:6px;">
                {'✅ High confidence' if confidence > 75 else '⚠️ Moderate — try redrawing larger'}
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence bar chart
            st.markdown("**Confidence per digit:**")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            colors = ['#c0392b' if i == pred_digit else '#2c7bb6' for i in range(10)]
            bars = ax.barh(range(10), probs * 100, color=colors, height=0.65, edgecolor='none')
            ax.set_yticks(range(10))
            ax.set_yticklabels([str(i) for i in range(10)], fontsize=11, fontweight='bold')
            ax.set_xlabel("Confidence (%)", fontsize=9)
            ax.set_xlim(0, 105)
            ax.set_facecolor('#f5f0e8')
            fig.patch.set_facecolor('#f5f0e8')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for bar, p in zip(bars, probs):
                if p > 0.01:
                    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                            f'{p*100:.1f}%', va='center', fontsize=8, color='#3a3530')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    elif predict_btn:
        st.info("Draw something on the canvas first!")
    else:
        st.markdown("""
        <div style="background:#f5f0e8;border:2px dashed #e8e2d6;border-radius:8px;
                    padding:40px;text-align:center;color:#8a7f72;">
          <div style="font-size:2.5rem;">✏️</div>
          <div style="margin-top:8px;font-size:0.9rem;">Draw a digit on the left, then click Predict</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── Sample MNIST images ───────────────────────────────────────
st.subheader("🖼️ Sample MNIST Training Images")
st.caption("These are actual images the model was trained on — 28×28 pixel grayscale images.")

fig2, axes = plt.subplots(2, 10, figsize=(14, 3))
fig2.patch.set_facecolor('#f5f0e8')
y_test_arr = np.array(y_test)
for digit in range(10):
    for row in range(2):
        idx = np.where(y_test_arr == digit)[0][row]
        axes[row, digit].imshow(X_test[idx].reshape(28, 28), cmap='gray')
        axes[row, digit].axis('off')
        if row == 0:
            axes[row, digit].set_title(str(digit), fontsize=11, fontweight='bold', color='#c0392b')
plt.tight_layout()
st.pyplot(fig2)
plt.close()

st.markdown("---")

# ── CNN Architecture explanation ──────────────────────────────
st.subheader("🧠 How the CNN Works")

a1, a2 = st.columns(2)

with a1:
    st.markdown("""
    <div class="concept-box">
      <h4>📥 Input Layer</h4>
      <p>Each image enters the network as a 28×28×1 tensor — 784 pixel values normalised between 0 and 1.</p>
    </div>
    <div class="concept-box">
      <h4>🔍 Conv2D Layers (×2)</h4>
      <p>Small filters (3×3) slide across the image learning low-level features like edges, curves, and strokes, then higher-level shapes.</p>
    </div>
    <div class="concept-box">
      <h4>⬇️ MaxPooling (×2)</h4>
      <p>Reduces spatial dimensions by keeping only the strongest activation in each 2×2 region. Makes the model translation-invariant.</p>
    </div>
    """, unsafe_allow_html=True)

with a2:
    st.markdown("""
    <div class="concept-box">
      <h4>🎲 Dropout (50%)</h4>
      <p>Randomly disables 50% of neurons during training. Prevents overfitting by forcing the network to learn robust features.</p>
    </div>
    <div class="concept-box">
      <h4>🧠 Dense Layer (128 neurons)</h4>
      <p>Fully connected layer that combines all extracted features into a rich representation before final classification.</p>
    </div>
    <div class="concept-box">
      <h4>📊 Softmax Output (10 neurons)</h4>
      <p>Converts raw scores into probabilities for each digit (0–9). All 10 values always sum to exactly 100%.</p>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#8a7f72;font-size:0.8rem;'>"
    "CSC 309 Mini Project · Task 5 · Handwritten Digit Recognition · "
    "Built with Streamlit + TensorFlow/Keras · MNIST Dataset"
    "</div>",
    unsafe_allow_html=True
)
