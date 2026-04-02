# ================================================================
#   CSC 309 — Task 5: Handwritten Digit Recognition
#   Web Application built with Streamlit + Scikit-learn
#   Dataset: MNIST | Model: Neural Network (MLPClassifier)
# ================================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

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
    font-family: 'Syne', sans-serif; font-size: 2.4rem;
    font-weight: 800; color: #1a1410; margin-bottom: 0;
  }
  .sub-title {
    font-size: 0.85rem; color: #8a7f72; letter-spacing: 2px;
    text-transform: uppercase; margin-top: 4px; margin-bottom: 28px;
  }
  .metric-card {
    background: #fdfaf4; border: 2px solid #1a1410; border-radius: 8px;
    padding: 18px 20px; box-shadow: 4px 4px 0 #1a1410; text-align: center;
  }
  .metric-num { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: #c0392b; }
  .metric-label { font-size: 0.75rem; color: #8a7f72; text-transform: uppercase; letter-spacing: 1.5px; }
  .pred-box {
    background: #fdfaf4; border: 3px solid #1a1410; border-radius: 10px;
    padding: 28px; box-shadow: 6px 6px 0 #c0392b; text-align: center;
  }
  .pred-number { font-family: 'Syne', sans-serif; font-size: 6rem; font-weight: 800; color: #c0392b; line-height: 1; }
  .pred-conf { font-size: 1.6rem; font-weight: 600; color: #1a1410; margin-top: 8px; }
  .concept-box {
    background: #fdfaf4; border: 2px solid #e8e2d6; border-radius: 8px;
    padding: 16px 20px; margin-bottom: 10px;
  }
  .concept-box h4 { color: #c0392b; margin-bottom: 4px; font-size: 0.9rem; }
  .concept-box p  { color: #3a3530; font-size: 0.85rem; line-height: 1.5; margin: 0; }
</style>
""", unsafe_allow_html=True)


# ── Load & train model (cached) ───────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)

    X_train, X_test = X[:18000], X[18000:20000]
    y_train, y_test = y[:18000], y[18000:20000]

    X_train = X_train / 255.0
    X_test  = X_test  / 255.0

    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        solver='adam',
        max_iter=30,
        random_state=42,
        verbose=False,
        early_stopping=True,
        validation_fraction=0.1,
    )
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    return model, acc, X_test, y_test


# ── Header ────────────────────────────────────────────────────
st.markdown('<div class="main-title">🔢 Handwritten Digit Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">CSC 309 · Task 5 · Neural Network · MNIST Dataset</div>', unsafe_allow_html=True)

with st.spinner("⏳ Training Neural Network on MNIST… (first load only, ~30–60 seconds)"):
    model, test_accuracy, X_test, y_test = load_model()

st.success(f"✅ Model ready! Test accuracy: **{test_accuracy*100:.2f}%**")

# ── Metrics ───────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-num">{test_accuracy*100:.1f}%</div><div class="metric-label">Test Accuracy</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><div class="metric-num">70K</div><div class="metric-label">MNIST Images</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><div class="metric-num">MLP</div><div class="metric-label">Model Type</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card"><div class="metric-num">10</div><div class="metric-label">Digit Classes</div></div>', unsafe_allow_html=True)

st.markdown("---")

# ── Draw & Predict ────────────────────────────────────────────
st.subheader("✏️ Draw a Digit & Get a Prediction")
st.caption("Draw any digit from 0–9 in the black box, then click **Predict**.")

col_draw, col_result = st.columns([1, 1], gap="large")

with col_draw:
    st.markdown("**Drawing Canvas**")
    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280, width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    predict_btn = st.button("🔍 Predict Digit", type="primary", use_container_width=True)
    st.caption("Tip: Draw thick and centred for best results.")

with col_result:
    st.markdown("**Prediction Result**")

    if predict_btn and canvas_result.image_data is not None:
        img_array = canvas_result.image_data

        if img_array[:, :, :3].sum() < 1000:
            st.warning("Canvas looks empty — please draw a digit first!")
        else:
            img = Image.fromarray(img_array.astype("uint8")).convert("L")
            img = img.resize((28, 28), Image.LANCZOS)
            img_np = np.array(img).astype("float32") / 255.0
            img_flat = img_np.flatten().reshape(1, -1)

            probs      = model.predict_proba(img_flat)[0]
            pred_digit = int(np.argmax(probs))
            confidence = float(probs[pred_digit]) * 100

            st.markdown(f"""
            <div class="pred-box">
              <div class="pred-number">{pred_digit}</div>
              <div class="pred-conf">{confidence:.1f}% confident</div>
              <div style="color:#8a7f72;font-size:0.8rem;margin-top:6px;">
                {'✅ High confidence' if confidence > 75 else '⚠️ Try redrawing larger and thicker'}
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Confidence per digit:**")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            colors = ['#c0392b' if i == pred_digit else '#2c7bb6' for i in range(10)]
            bars = ax.barh(range(10), probs * 100, color=colors, height=0.65, edgecolor='none')
            ax.set_yticks(range(10))
            ax.set_yticklabels([str(i) for i in range(10)], fontsize=11, fontweight='bold')
            ax.set_xlabel("Confidence (%)", fontsize=9)
            ax.set_xlim(0, 110)
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

# ── Sample images ─────────────────────────────────────────────
st.subheader("🖼️ Sample MNIST Test Images")
st.caption("Actual 28×28 pixel images the model was tested on.")

X_test_raw = (X_test * 255).astype("uint8")
fig2, axes = plt.subplots(2, 10, figsize=(14, 3))
fig2.patch.set_facecolor('#f5f0e8')
for digit in range(10):
    idxs = np.where(y_test == digit)[0]
    for row in range(2):
        if row < len(idxs):
            axes[row, digit].imshow(X_test_raw[idxs[row]].reshape(28, 28), cmap='gray')
        axes[row, digit].axis('off')
        if row == 0:
            axes[row, digit].set_title(str(digit), fontsize=11, fontweight='bold', color='#c0392b')
plt.tight_layout()
st.pyplot(fig2)
plt.close()

st.markdown("---")

# ── Confusion Matrix ──────────────────────────────────────────
st.subheader("📊 Confusion Matrix")
st.caption("Shows which digits the model confuses with each other.")

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

fig3, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xticklabels(range(10)); ax.set_yticklabels(range(10))
ax.set_xlabel("Predicted", fontsize=11); ax.set_ylabel("True", fontsize=11)
ax.set_title("Confusion Matrix — Which digits get confused?", fontsize=12, fontweight='bold')
for i in range(10):
    for j in range(10):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > cm.max()/2 else '#1a1410', fontsize=8)
plt.colorbar(im, ax=ax)
fig3.patch.set_facecolor('#f5f0e8')
plt.tight_layout()
st.pyplot(fig3)
plt.close()

st.markdown("---")

# ── Architecture explanation ──────────────────────────────────
st.subheader("🧠 How the Neural Network Works")

a1, a2 = st.columns(2)
with a1:
    st.markdown("""
    <div class="concept-box">
      <h4>📥 Input Layer (784 neurons)</h4>
      <p>Each 28×28 image is flattened into 784 pixel values, normalised between 0 and 1.</p>
    </div>
    <div class="concept-box">
      <h4>🔍 Hidden Layer 1 (256 neurons)</h4>
      <p>Learns low-level patterns like edges, curves, and strokes using ReLU activation.</p>
    </div>
    <div class="concept-box">
      <h4>🔍 Hidden Layer 2 (128 neurons)</h4>
      <p>Combines low-level features into higher-level digit representations.</p>
    </div>
    """, unsafe_allow_html=True)

with a2:
    st.markdown("""
    <div class="concept-box">
      <h4>📊 Output Layer (10 neurons)</h4>
      <p>One neuron per digit (0–9). Softmax converts scores into probabilities summing to 100%.</p>
    </div>
    <div class="concept-box">
      <h4>⚡ Adam Optimiser</h4>
      <p>Adaptively adjusts learning rates during training for fast, stable convergence.</p>
    </div>
    <div class="concept-box">
      <h4>🛑 Early Stopping</h4>
      <p>Training stops automatically when validation accuracy stops improving — prevents overfitting.</p>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#8a7f72;font-size:0.8rem;'>"
    "CSC 309 Mini Project · Task 5 · Handwritten Digit Recognition · "
    "Built with Streamlit + Scikit-learn · MNIST Dataset"
    "</div>", unsafe_allow_html=True
)
