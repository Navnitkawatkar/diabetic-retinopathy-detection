import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DR Detection System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main { background-color: #f0f4f8; }

    .header-box {
        background: linear-gradient(135deg, #1a3c5e 0%, #1a8fa8 100%);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 25px;
        text-align: center;
        color: white;
    }

    .header-box h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }

    .header-box p {
        font-size: 1rem;
        opacity: 0.85;
        margin: 8px 0 0 0;
    }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border-top: 4px solid #1a8fa8;
    }

    .metric-card h2 {
        font-size: 2rem;
        font-weight: 700;
        color: #1a3c5e;
        margin: 0;
    }

    .metric-card p {
        color: #666;
        font-size: 0.85rem;
        margin: 5px 0 0 0;
    }

    .result-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-top: 20px;
    }

    .grade-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 25px;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 10px;
    }

    .upload-box {
        background: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }

    .sidebar-info {
        background: #1a3c5e;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #1a3c5e, #1a8fa8);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
    }

    .footer {
        text-align: center;
        color: #888;
        font-size: 0.8rem;
        margin-top: 40px;
        padding: 20px;
        border-top: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

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
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div class='sidebar-info'>
            <h3 style='margin:0; color:white;'>👁️ DR Detection</h3>
            <p style='margin:5px 0 0 0; opacity:0.8; font-size:0.85rem;'>
                 Retinal analysis
            </p>
        </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", 
                    ["🏠 Home", "🔍 Diagnose", "📊 About DR", "ℹ️ About Project"],
                    label_visibility="hidden")

    st.markdown("---")
    st.markdown("### 📋 Patient Info")
    patient_name = st.text_input("Patient Name", placeholder="Enter name")
    patient_age  = st.number_input("Age", min_value=1, max_value=120, value=30)
    diabetes_type = st.selectbox("Diabetes Type", 
                                  ["Type 1", "Type 2", "Gestational", "Unknown"])
    st.markdown("---")
    st.markdown("""
        <div style='font-size:0.75rem; color:#888; text-align:center;'>
            Minor Project | B.Tech CSE<br>
            Diabetic Retinopathy Detection
        </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
    <div class='header-box'>
        <h1>👁️ Diabetic Retinopathy Detection System</h1>
        <p>Upload a retinal fundus image for instant  DR grade classification</p>
    </div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────

# ── HOME PAGE ──
if page == "🏠 Home":
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
            <div class='metric-card'>
                <h2>11.7M</h2>
                <p>Model Parameters</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class='metric-card'>
                <h2>0.85</h2>
                <p>AUC-ROC Score</p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class='metric-card'>
                <h2>60%</h2>
                <p>Validation Accuracy</p>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
            <div class='metric-card'>
                <h2>2750</h2>
                <p>Training Images</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 What is Diabetic Retinopathy?")
        st.markdown("""
        Diabetic Retinopathy (DR) is a diabetes complication that affects the eyes.
        It's caused by damage to blood vessels in the retina and is a leading cause
        of blindness worldwide.

        **Key Facts:**
        - Affects **1 in 3** diabetic patients
        - Over **100 million** people affected globally
        - Early detection can prevent **90%** of blindness cases
        - Regular screening is critical for all diabetic patients
        """)

    with col2:
        st.markdown("### 🤖 How Does This System Work?")
        st.markdown("""
        This system uses **Deep Learning** to automatically analyze retinal
        fundus photographs and classify DR severity.

        **Technology:**
        - 🧠 EfficientNetB3 Transfer Learning
        - 🔧 Ben Graham Image Preprocessing
        - 📊 5-Class Grade Classification
        - ⚡ Real-time Prediction < 2 seconds
        """)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 Go to **Diagnose** in the sidebar to analyze a retinal image!")

# ── DIAGNOSE PAGE ──
elif page == "🔍 Diagnose":
    st.markdown("### 📤 Upload Retinal Fundus Image")

    uploaded_file = st.file_uploader(
        "Choose a retinal fundus image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original Image**")
            st.image(image, use_column_width=True)
        with col2:
            st.markdown("**Preprocessed Image**")
            preprocessed = preprocess_image(image)
            st.image(preprocessed, use_column_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🔍 Analyze Image"):
            with st.spinner("Analyzing retinal image..."):
                model = load_dr_model()
                img_batch = np.expand_dims(preprocessed, axis=0)
                probs = model.predict(img_batch, verbose=0)[0]
                grade = int(np.argmax(probs))
                confidence = float(probs[grade]) * 100

            st.markdown("<hr>", unsafe_allow_html=True)

            # Patient report header
            if patient_name:
                st.markdown(f"### 🏥 Diagnosis Report — {patient_name}")
                col1, col2, col3 = st.columns(3)
                col1.info(f"👤 **Patient:** {patient_name}")
                col2.info(f"🎂 **Age:** {patient_age}")
                col3.info(f"🩺 **Diabetes:** {diabetes_type}")

            # Result
            st.markdown(f"""
                <div class='result-card'>
                    <span class='grade-badge' style='background:{CLASS_COLORS[grade]}22;
                          color:{CLASS_COLORS[grade]}; border: 2px solid {CLASS_COLORS[grade]};'>
                        {CLASS_EMOJI[grade]} {CLASS_NAMES[grade]}
                    </span>
                    <h3 style='margin:10px 0 5px 0;'>Confidence: {confidence:.1f}%</h3>
                    <p style='color:#555;'>💊 {ADVICE[grade]}</p>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Confidence chart
            st.markdown("### 📊 Probability Distribution")
            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.barh(CLASS_NAMES, probs * 100,
                          color=CLASS_COLORS, edgecolor='white', height=0.6)
            ax.set_xlabel("Probability (%)")
            ax.set_xlim(0, 100)
            for bar, prob in zip(bars, probs):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                       f'{prob*100:.1f}%', va='center', fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)

            # Alert
            if grade >= 3:
                st.error("🚨 Severe DR detected! Please consult an ophthalmologist immediately.")
            elif grade == 2:
                st.warning("⚠️ Moderate DR detected. Please schedule an eye exam soon.")
            else:
                st.success("✅ Low risk detected. Continue regular annual checkups.")
    else:
        st.info("👆 Upload a retinal fundus image above to get started")

# ── ABOUT DR PAGE ──
elif page == "📊 About DR":
    st.markdown("### 📊 Diabetic Retinopathy Grading Scale")

    for i, (name, color, emoji, advice) in enumerate(
            zip(CLASS_NAMES, CLASS_COLORS, CLASS_EMOJI, ADVICE)):
        st.markdown(f"""
            <div style='border-left:5px solid {color}; padding:15px 20px;
                        margin-bottom:12px; background:{color}11; border-radius:8px;'>
                <h4 style='margin:0; color:{color};'>{emoji} Grade {i} — {name}</h4>
                <p style='margin:5px 0 0 0; color:#555;'>{advice}</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📈 Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        metrics = {
            "Accuracy": 60,
            "AUC Score": 85,
            "Precision": 80,
            "Recall": 32
        }
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(metrics.keys(), metrics.values(),
                     color=["#1a8fa8", "#1a3c5e", "#2ecc71", "#f39c12"],
                     edgecolor='white')
        ax.set_ylabel("Score (%)")
        ax.set_ylim(0, 100)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 1,
                   f'{bar.get_height()}%',
                   ha='center', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("""
        | Metric | Score |
        |--------|-------|
        | Accuracy | 60% |
        | AUC-ROC | 0.85 |
        | Precision | 80% |
        | Parameters | 11.7M |
        | Training Images | 2750 |
        | Inference Time | < 2s |
        """)

# ── ABOUT PROJECT PAGE ──
elif page == "ℹ️ About Project":
    st.markdown("### ℹ️ About This Project")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Project Title:**
        Diabetic Retinopathy Detection via Retinal Fundus Images

        **Technology Stack:**
        - 🐍 Python 3.11
        - 🧠 TensorFlow 2.15 / Keras
        - 👁️ OpenCV (Image Processing)
        - 🌐 Streamlit (Web App)
        - 📊 EfficientNetB3 (Deep Learning)

        **Dataset:**
        - Source: Kaggle DR Dataset
        - Total Images: 2,750
        - Classes: 5 (Grade 0-4)
        """)
    with col2:
        st.markdown("""
        **Model Architecture:**
        - Input: 224×224×3
        - Backbone: EfficientNetB3
        - GlobalAveragePooling2D
        - Dense(512) + Dropout(0.4)
        - Dense(256) + Dropout(0.3)
        - Output: Softmax (5 classes)

        **Training:**
        - Phase 1: Frozen backbone (5 epochs)
        - Phase 2: Fine-tuning (20 epochs)
        - Optimizer: Adam
        - Loss: Categorical Crossentropy
        """)

    st.markdown("---")
    st.markdown("""
        <div style='text-align:center; padding:20px;
                    background:#1a3c5e; border-radius:10px; color:white;'>
            <h3 style='margin:0;'>Minor Project Submission</h3>
            <p style='margin:8px 0 0 0; opacity:0.8;'>
                B.Tech Computer Science Engineering
            </p>
        </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
    <div class='footer'>
        👁️ Diabetic Retinopathy Detection System |
        Minor Project | B.Tech CSE |
        Built with TensorFlow & Streamlit
    </div>
""", unsafe_allow_html=True)
