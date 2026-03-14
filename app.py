import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="👁️",
    layout="centered"
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
IMG_SIZE = 224
CLASS_NAMES = [
    "No DR (Grade 0)",
    "Mild DR (Grade 1)",
    "Moderate DR (Grade 2)",
    "Severe DR (Grade 3)",
    "Proliferative DR (Grade 4)"
]
CLASS_COLORS = ["#2ecc71", "#27ae60", "#f39c12", "#e67e22", "#e74c3c"]
CLASS_EMOJI  = ["✅", "🟡", "🟠", "🔴", "🆘"]
ADVICE = [
    "No treatment needed. Continue regular checkups every year.",
    "Mild changes detected. Consult an ophthalmologist within 6 months.",
    "Moderate DR detected. See an ophthalmologist within 3 months.",
    "Severe DR detected. Urgent ophthalmologist visit required!",
    "Proliferative DR! Immediate medical attention required!"
]

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_dr_model():
    import tensorflow as tf
    model = tf.keras.models.load_model("dr_model_saved")
    return model

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_image(img):
    img = np.array(img.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.addWeighted(
        img, 4,
        cv2.GaussianBlur(img, (0, 0), IMG_SIZE / 30), -4,
        128
    )
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask,
               (img.shape[1] // 2, img.shape[0] // 2),
               int(min(img.shape[:2]) * 0.45),
               1, -1)
    img = img * mask[:, :, np.newaxis]
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return img

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

# Header
st.markdown("""
    <h1 style='text-align:center; color:#1a5276;'>
        👁️ Diabetic Retinopathy Detection
    </h1>
    <p style='text-align:center; color:gray; font-size:16px;'>
         retinal fundus image analysis using EfficientNetB3
    </p>
    <hr>
""", unsafe_allow_html=True)

# Info cards
col1, col2, col3 = st.columns(3)
col1.metric("🧠 Model", "EfficientNetB3")
col2.metric("📊 AUC Score", "0.85")
col3.metric("🖼️ Input Size", "224 × 224")

st.markdown("<hr>", unsafe_allow_html=True)

# Upload section
st.markdown("### 📤 Upload Retinal Fundus Image")
uploaded_file = st.file_uploader(
    "Upload a retinal fundus image (JPG, PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Image**")
        st.image(image, use_column_width=True)

    with col2:
        st.markdown("**Preprocessed Image**")
        preprocessed = preprocess_image(image)
        st.image(preprocessed, use_column_width=True)

    # Predict button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 Analyze Image", use_container_width=True):
        with st.spinner("Analyzing retinal image..."):
            # Load model and predict
            model = load_dr_model()
            img_batch = np.expand_dims(preprocessed, axis=0)
            probs = model.predict(img_batch, verbose=0)[0]
            grade = int(np.argmax(probs))
            confidence = float(probs[grade]) * 100

        # Result
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 🩺 Diagnosis Result")

        # Main result card
        st.markdown(f"""
            <div style='
                background-color:{CLASS_COLORS[grade]}22;
                border-left: 6px solid {CLASS_COLORS[grade]};
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
            '>
                <h2 style='color:{CLASS_COLORS[grade]}; margin:0;'>
                    {CLASS_EMOJI[grade]} {CLASS_NAMES[grade]}
                </h2>
                <p style='font-size:18px; margin:8px 0 0 0;'>
                    Confidence: <strong>{confidence:.1f}%</strong>
                </p>
                <p style='color:#555; margin:8px 0 0 0;'>
                    💊 {ADVICE[grade]}
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Probability bars
        st.markdown("### 📊 Probability Distribution")
        for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.progress(float(prob), text=name)
            with col2:
                st.markdown(f"**{prob*100:.1f}%**")

        # Warning for severe cases
        if grade >= 3:
            st.error("⚠️ Severe DR detected! Please consult an ophthalmologist immediately.")
        elif grade == 2:
            st.warning("⚠️ Moderate DR detected. Please schedule an eye exam soon.")
        else:
            st.success("✅ Low risk detected. Continue regular annual checkups.")

else:
    # Placeholder when no image uploaded
    st.info("👆 Upload a retinal fundus image above to get started")

    # Sample info
    st.markdown("### 📋 DR Grading Scale")
    for i, (name, color, emoji, advice) in enumerate(zip(CLASS_NAMES, CLASS_COLORS, CLASS_EMOJI, ADVICE)):
        st.markdown(f"""
            <div style='
                border-left: 4px solid {color};
                padding: 8px 15px;
                margin-bottom: 8px;
                background:{color}11;
                border-radius:5px;
            '>
                <strong>{emoji} Grade {i} — {name}</strong><br>
                <small style='color:#666;'>{advice}</small>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
    <p style='text-align:center; color:gray; font-size:13px;'>
        Minor Project | B.Tech CSE | Diabetic Retinopathy Detection via Retinal Fundus Images
    </p>
""", unsafe_allow_html=True)

