import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import csv

# ─────────────────────────────────────────────
# PAGE CONFIG  (#7)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DR Screening",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main { background-color: #f7f9fc; }

    /* ── Header ── */
    .header-box {
        background: #0b1f3a;
        padding: 36px 40px;
        border-radius: 16px;
        margin-bottom: 28px;
        display: flex;
        align-items: center;
        gap: 20px;
    }
    .header-box .icon {
        font-size: 3rem;
        line-height: 1;
    }
    .header-box h1 {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        font-weight: 400;
        color: #ffffff;
        margin: 0 0 4px 0;
    }
    .header-box p {
        font-size: 0.9rem;
        color: #8faec7;
        margin: 0;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 16px;
        text-align: center;
        border: 1px solid #e8edf3;
        border-top: 3px solid #1a6fa8;
    }
    .metric-card .val {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        color: #0b1f3a;
        line-height: 1;
    }
    .metric-card .lbl {
        color: #7a8fa6;
        font-size: 0.8rem;
        margin-top: 6px;
    }

    /* ── Upload zone ── */
    .upload-zone {
        background: white;
        border: 2px dashed #a8c4e0;
        border-radius: 16px;
        padding: 36px 24px;
        text-align: center;
        transition: border-color 0.2s;
    }
    .upload-zone:hover { border-color: #1a6fa8; }
    .upload-zone .uz-icon { font-size: 2.5rem; margin-bottom: 10px; }
    .upload-zone .uz-title { font-size: 1rem; font-weight: 500; color: #0b1f3a; }
    .upload-zone .uz-sub { font-size: 0.82rem; color: #7a8fa6; margin-top: 4px; }

    /* ── Result card ── */
    .result-card {
        background: white;
        border-radius: 16px;
        padding: 28px;
        border: 1px solid #e8edf3;
        margin-top: 20px;
    }

    /* ── Severity badge ── */
    .grade-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 18px;
        border-radius: 30px;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 12px;
    }

    /* ── Clinical advice box ── */
    .advice-box {
        border-left: 4px solid;
        padding: 12px 16px;
        border-radius: 0 10px 10px 0;
        margin-top: 14px;
        font-size: 0.9rem;
    }

    /* ── Progress bars ── */
    .prob-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 8px;
        font-size: 0.84rem;
    }
    .prob-label { width: 180px; color: #3a4f66; flex-shrink: 0; }
    .prob-bar-wrap {
        flex: 1;
        background: #eef2f7;
        border-radius: 6px;
        height: 10px;
        overflow: hidden;
    }
    .prob-bar { height: 10px; border-radius: 6px; }
    .prob-pct { width: 44px; text-align: right; color: #3a4f66; font-weight: 500; }

    /* ── Image caption ── */
    .img-caption {
        font-size: 0.78rem;
        color: #7a8fa6;
        text-align: center;
        margin-top: 6px;
    }

    /* ── Section label ── */
    .section-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #7a8fa6;
        margin-bottom: 10px;
    }

    /* ── Patient info pills ── */
    .patient-pill {
        display: inline-block;
        background: #eef4fb;
        color: #1a6fa8;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.83rem;
        font-weight: 500;
        margin-right: 8px;
        margin-bottom: 8px;
    }

    /* ── Sidebar ── */
    .sidebar-header {
        background: #0b1f3a;
        color: white;
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 16px;
    }
    .sidebar-header h3 { margin: 0; font-size: 1rem; font-weight: 500; }
    .sidebar-header p  { margin: 4px 0 0 0; font-size: 0.8rem; color: #8faec7; }

    .disclaimer-box {
        background: #fff8e1;
        border: 1px solid #ffe082;
        border-radius: 10px;
        padding: 12px 14px;
        font-size: 0.78rem;
        color: #6d4c00;
        margin-top: 12px;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: #0b1f3a;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 28px;
        font-size: 0.95rem;
        font-weight: 500;
        width: 100%;
        transition: background 0.2s;
    }
    .stButton > button:hover { background: #1a3a6a; }

    /* ── Grade scale rows (About DR) ── */
    .grade-row {
        border-left: 5px solid;
        padding: 14px 18px;
        border-radius: 0 10px 10px 0;
        margin-bottom: 10px;
    }
    .grade-row h4 { margin: 0 0 4px 0; font-size: 0.95rem; font-weight: 600; }
    .grade-row p  { margin: 0; font-size: 0.85rem; color: #555; }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: #aab6c5;
        font-size: 0.78rem;
        margin-top: 48px;
        padding: 20px;
        border-top: 1px solid #e4eaf2;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
IMG_SIZE = 224

CLASS_NAMES  = ["No DR (Grade 0)", "Mild DR (Grade 1)", "Moderate DR (Grade 2)",
                "Severe DR (Grade 3)", "Proliferative DR (Grade 4)"]
CLASS_COLORS = ["#2ecc71", "#27ae60", "#f39c12", "#e67e22", "#e74c3c"]
CLASS_BG     = ["#eafaf1", "#eafaf1", "#fef9e7", "#fef5ec", "#fdf0f0"]
CLASS_EMOJI  = ["✅", "🟡", "🟠", "🔴", "🆘"]

CLINICAL_INFO = [
    {
        "title": "No diabetic retinopathy",
        "desc":  "No visible signs of retinal damage. Blood vessels appear healthy.",
        "advice": "Continue annual dilated eye exams. Maintain good blood sugar control.",
        "action": "Annual checkup",
        "border": "#2ecc71"
    },
    {
        "title": "Mild non-proliferative DR",
        "desc":  "Small areas of balloon-like swelling (microaneurysms) in retinal blood vessels.",
        "advice": "Consult an ophthalmologist within 6 months. Monitor blood glucose and blood pressure.",
        "action": "6-month follow-up",
        "border": "#27ae60"
    },
    {
        "title": "Moderate non-proliferative DR",
        "desc":  "Some blood vessels that nourish the retina are blocked. Hard exudates may be present.",
        "advice": "See an ophthalmologist within 3 months. Referral for fluorescein angiography may be needed.",
        "action": "3-month referral",
        "border": "#f39c12"
    },
    {
        "title": "Severe non-proliferative DR",
        "desc":  "Many more blood vessels are blocked, depriving the retina of blood supply.",
        "advice": "Urgent ophthalmologist visit required. High risk of progression to proliferative DR.",
        "action": "Urgent referral",
        "border": "#e67e22"
    },
    {
        "title": "Proliferative DR",
        "desc":  "New fragile blood vessels grow on the retina. Risk of vitreous hemorrhage and retinal detachment.",
        "advice": "Immediate medical attention required. Laser treatment or anti-VEGF injections may be indicated.",
        "action": "Immediate care",
        "border": "#e74c3c"
    }
]

# ─────────────────────────────────────────────
# SESSION STATE  (#6)
# ─────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "preprocessed" not in st.session_state:
    st.session_state.preprocessed = None
if "file_meta" not in st.session_state:
    st.session_state.file_meta = {}

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_dr_model():
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
# HELPERS
# ─────────────────────────────────────────────
def human_size(n_bytes):
    if n_bytes < 1024:
        return f"{n_bytes} B"
    elif n_bytes < 1024 ** 2:
        return f"{n_bytes/1024:.1f} KB"
    return f"{n_bytes/1024**2:.1f} MB"

def confidence_bars(probs):
    """Render inline HTML confidence bars for all 5 classes."""
    html = ""
    for i, (name, prob, color) in enumerate(zip(CLASS_NAMES, probs, CLASS_COLORS)):
        pct = prob * 100
        html += f"""
        <div class="prob-row">
            <span class="prob-label">{CLASS_EMOJI[i]} {name}</span>
            <div class="prob-bar-wrap">
                <div class="prob-bar" style="width:{pct:.1f}%; background:{color};"></div>
            </div>
            <span class="prob-pct">{pct:.1f}%</span>
        </div>
        """
    return html

def make_csv(patient_name, patient_age, diabetes_type, filename, grade, probs):
    """Build a CSV bytes object for download."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Field", "Value"])
    writer.writerow(["Patient Name", patient_name or "N/A"])
    writer.writerow(["Age", patient_age])
    writer.writerow(["Diabetes Type", diabetes_type])
    writer.writerow(["Image File", filename])
    writer.writerow(["Predicted Grade", grade])
    writer.writerow(["Predicted Label", CLASS_NAMES[grade]])
    writer.writerow([])
    writer.writerow(["Class", "Probability (%)"])
    for name, prob in zip(CLASS_NAMES, probs):
        writer.writerow([name, f"{prob*100:.2f}"])
    return output.getvalue().encode()

# ─────────────────────────────────────────────
# SIDEBAR  (#8)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div class='sidebar-header'>
            <h3>🔬 DR Screening System</h3>
            <p>EfficientNetB3 · 5-class grading</p>
        </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation",
                    ["🏠 Home", "🔍 Diagnose", "📊 About DR", "ℹ️ About Project"],
                    label_visibility="hidden")

    st.markdown("---")
    st.markdown("**Patient Information**")
    patient_name  = st.text_input("Patient Name", placeholder="Enter name")
    patient_age   = st.number_input("Age", min_value=1, max_value=120, value=30)
    diabetes_type = st.selectbox("Diabetes Type",
                                 ["Type 1", "Type 2", "Gestational", "Unknown"])

    st.markdown("---")
    st.markdown("**Model Info**")
    st.caption("Architecture: EfficientNetB3")
    st.caption("Dataset: Kaggle DR (2,750 images)")
    st.caption("AUC-ROC: 0.85 · Accuracy: 60%")
    st.caption("Inference time: < 2 seconds")

    st.markdown("""
        <div class='disclaimer-box'>
            ⚠️ <strong>Research use only.</strong> This tool is not a substitute
            for clinical diagnosis by a qualified ophthalmologist.
        </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
    <div class='header-box'>
        <div class='icon'>👁️</div>
        <div>
            <h1>Diabetic Retinopathy Detection</h1>
            <p>Upload a retinal fundus image for instant DR grade classification · EfficientNetB3</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────
if page == "🏠 Home":
    col1, col2, col3, col4 = st.columns(4)
    for col, val, lbl in zip(
        [col1, col2, col3, col4],
        ["11.7M", "0.85", "60%", "2,750"],
        ["Model Parameters", "AUC-ROC Score", "Validation Accuracy", "Training Images"]
    ):
        with col:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='val'>{val}</div>
                    <div class='lbl'>{lbl}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🎯 What is Diabetic Retinopathy?")
        st.markdown("""
        Diabetic Retinopathy (DR) is a diabetes complication that damages the blood
        vessels in the retina. It is a leading cause of blindness worldwide, yet
        early detection can prevent **90% of blindness cases**.

        - Affects **1 in 3** diabetic patients
        - Over **100 million** people affected globally
        - Regular screening is critical for all diabetic patients
        """)
    with col2:
        st.markdown("#### 🤖 How Does This System Work?")
        st.markdown("""
        This system uses **Deep Learning** to automatically analyze retinal
        fundus photographs and classify DR severity into 5 grades.

        - 🧠 EfficientNetB3 Transfer Learning
        - 🔧 Ben Graham image preprocessing
        - 📊 5-class grade classification (0–4)
        - ⚡ Real-time prediction in under 2 seconds
        """)

    st.info("👈 Open **Diagnose** in the sidebar to analyse a retinal image.")

# ─────────────────────────────────────────────
# DIAGNOSE PAGE
# ─────────────────────────────────────────────
elif page == "🔍 Diagnose":

    # ── Reset button (#10) ──
    if st.session_state.result is not None:
        if st.button("🔄 Analyse another image"):
            st.session_state.result = None
            st.session_state.uploaded_image = None
            st.session_state.preprocessed = None
            st.session_state.file_meta = {}
            st.rerun()

    # ── Upload zone (#1) ──
    if st.session_state.uploaded_image is None:
        st.markdown("""
            <div class='upload-zone'>
                <div class='uz-icon'>🩺</div>
                <div class='uz-title'>Drag a fundus image here, or click to browse</div>
                <div class='uz-sub'>Accepted formats: JPG · JPEG · PNG &nbsp;|&nbsp; Max recommended: 10 MB</div>
            </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload retinal fundus image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.session_state.uploaded_image = image
            st.session_state.file_meta = {
                "name": uploaded_file.name,
                "size": human_size(uploaded_file.size)
            }
            st.session_state.preprocessed = preprocess_image(image)
            st.rerun()

    else:
        image       = st.session_state.uploaded_image
        preprocessed = st.session_state.preprocessed
        meta        = st.session_state.file_meta

        # ── Image preview (#2) ──
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='section-label'>Original image</div>", unsafe_allow_html=True)
            st.image(image, use_column_width=True)
            st.markdown(f"<div class='img-caption'>{meta.get('name','')} · {meta.get('size','')}</div>",
                        unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='section-label'>Preprocessed (Ben Graham)</div>", unsafe_allow_html=True)
            st.image(preprocessed, use_column_width=True)
            st.markdown("<div class='img-caption'>Contrast-enhanced · circular crop · 224×224</div>",
                        unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Analyse button ──
        if st.session_state.result is None:
            if st.button("🔬 Analyse image"):
                with st.spinner("Analysing retinal image — please wait…"):   # (#6)
                    model    = load_dr_model()
                    img_batch = np.expand_dims(preprocessed, axis=0)
                    probs    = model.predict(img_batch, verbose=0)[0]
                    grade    = int(np.argmax(probs))
                    st.session_state.result = {"probs": probs, "grade": grade}
                st.rerun()

        # ── Results ──
        if st.session_state.result is not None:
            probs = st.session_state.result["probs"]
            grade = st.session_state.result["grade"]
            info  = CLINICAL_INFO[grade]
            confidence = float(probs[grade]) * 100

            st.markdown("<hr>", unsafe_allow_html=True)

            # Patient pills
            if patient_name:
                st.markdown(f"#### 🏥 Diagnosis Report — {patient_name}")
                st.markdown(f"""
                    <span class='patient-pill'>👤 {patient_name}</span>
                    <span class='patient-pill'>🎂 Age {patient_age}</span>
                    <span class='patient-pill'>🩺 {diabetes_type}</span>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

            # ── Severity badge (#3) ──
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("<div class='section-label'>Diagnosis</div>", unsafe_allow_html=True)
                st.markdown(f"""
                    <div class='result-card'>
                        <span class='grade-badge'
                              style='background:{CLASS_BG[grade]};
                                     color:{CLASS_COLORS[grade]};
                                     border:2px solid {CLASS_COLORS[grade]};'>
                            {CLASS_EMOJI[grade]} {CLASS_NAMES[grade]}
                        </span>
                        <div style='font-size:0.9rem; color:#3a4f66; margin-bottom:6px;'>
                            Confidence: <strong>{confidence:.1f}%</strong>
                        </div>
                        <div style='font-size:0.88rem; color:#555; margin-bottom:10px;'>
                            {info["desc"]}
                        </div>
                        <div class='advice-box'
                             style='border-color:{info["border"]}; background:{CLASS_BG[grade]};
                                    color:#3a4f66;'>
                            💊 {info["advice"]}
                        </div>
                        <div style='margin-top:12px; display:inline-block;
                                    background:{CLASS_BG[grade]}; color:{CLASS_COLORS[grade]};
                                    border-radius:20px; padding:4px 14px; font-size:0.8rem;
                                    font-weight:600; border:1px solid {CLASS_COLORS[grade]};'>
                            Recommended action: {info["action"]}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            # ── Confidence bars (#4) ──
            with col2:
                st.markdown("<div class='section-label'>Confidence by grade</div>", unsafe_allow_html=True)
                st.markdown(f"""
                    <div class='result-card' style='padding-top:20px;'>
                        {confidence_bars(probs)}
                    </div>
                """, unsafe_allow_html=True)

            # ── Alert banners ──
            st.markdown("<br>", unsafe_allow_html=True)
            if grade >= 3:
                st.error("🚨 **Severe DR detected.** Please consult an ophthalmologist immediately.")
            elif grade == 2:
                st.warning("⚠️ **Moderate DR detected.** Please schedule an eye examination soon.")
            else:
                st.success("✅ **Low-risk result.** Continue regular annual screenings.")

            # ── Download CSV (#9) ──
            csv_bytes = make_csv(
                patient_name, patient_age, diabetes_type,
                meta.get("name", "unknown"), grade, probs
            )
            st.download_button(
                label="⬇️ Download results as CSV",
                data=csv_bytes,
                file_name=f"dr_result_{meta.get('name','scan').split('.')[0]}.csv",
                mime="text/csv"
            )

# ─────────────────────────────────────────────
# ABOUT DR PAGE  (#5 — clinical descriptions)
# ─────────────────────────────────────────────
elif page == "📊 About DR":
    st.markdown("### Diabetic Retinopathy Grading Scale")
    st.caption("International Clinical DR Disease Severity Scale (ICDRS)")

    for i, (info, color, emoji) in enumerate(zip(CLINICAL_INFO, CLASS_COLORS, CLASS_EMOJI)):
        st.markdown(f"""
            <div class='grade-row' style='border-color:{color}; background:{CLASS_BG[i]};'>
                <h4 style='color:{color};'>{emoji} Grade {i} — {info["title"]}</h4>
                <p>{info["desc"]}</p>
                <p style='margin-top:6px; font-size:0.82rem; color:#666;'>
                    <strong>Clinical action:</strong> {info["advice"]}
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Model Performance Metrics")

    col1, col2 = st.columns(2)
    with col1:
        metrics = {"Accuracy": 60, "AUC-ROC": 85, "Precision": 80, "Recall": 32}
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(metrics.keys(), metrics.values(),
                      color=["#1a6fa8", "#0b1f3a", "#2ecc71", "#f39c12"],
                      edgecolor="white", width=0.5)
        ax.set_ylabel("Score (%)")
        ax.set_ylim(0, 100)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5,
                    f"{bar.get_height()}%",
                    ha="center", fontsize=10, color="#333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("#f7f9fc")
        fig.patch.set_facecolor("#f7f9fc")
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("""
        | Metric | Value |
        |--------|-------|
        | Accuracy | 60% |
        | AUC-ROC | 0.85 |
        | Precision | 80% |
        | Recall | 32% |
        | Parameters | 11.7M |
        | Training images | 2,750 |
        | Inference time | < 2 s |
        """)

# ─────────────────────────────────────────────
# ABOUT PROJECT PAGE
# ─────────────────────────────────────────────
elif page == "ℹ️ About Project":
    st.markdown("### About This Project")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Project Title:**
        Diabetic Retinopathy Detection via Retinal Fundus Images

        **Technology Stack:**
        - 🐍 Python 3.11
        - 🧠 TensorFlow 2.15 / Keras
        - 👁️ OpenCV (image processing)
        - 🌐 Streamlit (web app)
        - 📊 EfficientNetB3 (deep learning)
        - 📂 Kaggle DR Dataset
        """)

    with col2:
        st.markdown("""
        **Model Architecture:**
        - Input: 224 × 224 × 3
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
        <div style='text-align:center; padding:24px; background:#0b1f3a;
                    border-radius:12px; color:white;'>
            <div style='font-family:"DM Serif Display",serif; font-size:1.3rem;
                        font-weight:400; margin-bottom:6px;'>
                Minor Project Submission
            </div>
            <div style='opacity:0.7; font-size:0.88rem;'>
                B.Tech Computer Science Engineering · Diabetic Retinopathy Detection
            </div>
        </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
    <div class='footer'>
        🔬 Diabetic Retinopathy Detection System &nbsp;·&nbsp;
    </div>
""", unsafe_allow_html=True)