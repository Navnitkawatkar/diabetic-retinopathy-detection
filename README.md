# 👁️ Diabetic Retinopathy Detection via Retinal Fundus Images

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://navnitkawatkar-diabetic-retinopathy-detection.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🔗 Live Demo
👉 **[Click here to open the Web App](https://navnitkawatkar-diabetic-retinopathy-detection.streamlit.app)**

---

## 📌 About the Project
Diabetic Retinopathy (DR) is a leading cause of preventable blindness,
affecting over 100 million diabetic patients worldwide. This project
presents an **automated DR detection system** using Deep Learning on
retinal fundus photographs.

The system classifies retinal images into **5 severity grades** using
**Transfer Learning with EfficientNetB3** as the backbone model.

---

## 🏆 Model Performance
| Metric | Score |
|--------|-------|
| Validation Accuracy | 60% |
| AUC-ROC Score | 0.85 |
| Precision | 80% |
| Model Size | 44.66 MB |
| Inference Time | < 2 seconds |

---

## 📊 DR Grading Scale
| Grade | Severity | Description |
|-------|----------|-------------|
| 0 | No DR | No visible pathological changes |
| 1 | Mild DR | Microaneurysms only |
| 2 | Moderate DR | Hemorrhages and hard exudates |
| 3 | Severe DR | Cotton wool spots, venous beading |
| 4 | Proliferative DR | Neovascularization — urgent treatment needed |

---

## 🧠 Model Architecture
```
Input (224×224×3)
      ↓
EfficientNetB3 Backbone (pretrained ImageNet)
      ↓
GlobalAveragePooling2D
      ↓
Dense(512) + BatchNorm + Dropout(0.4)
      ↓
Dense(256) + BatchNorm + Dropout(0.3)
      ↓
Output: Softmax (5 classes)
```

**Total Parameters:** 11,706,164 (44.66 MB)
**Trainable Parameters:** 921,093 (3.51 MB)

---

## 🔧 Preprocessing Pipeline
```
Raw Fundus Image
      ↓
Ben Graham Contrast Enhancement
      ↓
Circular Crop (remove black borders)
      ↓
Resize to 224×224
      ↓
Normalize [0, 1]
```

---

## 🚀 Technology Stack
| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| Deep Learning | TensorFlow 2.15 / Keras |
| Model Backbone | EfficientNetB3 |
| Image Processing | OpenCV |
| Data Analysis | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Dataset | Kaggle DR Dataset (2750 images) |

---

## 📁 Project Structure
```
diabetic-retinopathy-detection/
├── dataset/
│   ├── 0/          ← No DR images
│   ├── 1/          ← Mild DR images
│   ├── 2/          ← Moderate DR images
│   ├── 3/          ← Severe DR images
│   └── 4/          ← Proliferative DR images
├── dr_detection.py     ← Main model training code
├── app.py              ← Streamlit web application
├── dr_model.h5         ← Trained model weights
├── requirements.txt    ← Python dependencies
├── training_history.png← Accuracy & loss graphs
└── confusion_matrix.png← Evaluation heatmap
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Navnitkawatkar/diabetic-retinopathy-detection.git
cd diabetic-retinopathy-detection
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate.bat
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the web app
```bash
streamlit run app.py
```

### 5. Train the model (optional)
```bash
python dr_detection.py
```

---

## 📦 Dataset
- **Source:** Kaggle Diabetic Retinopathy Dataset
- **Total Images:** 2,750
- **Classes:** 5 (Grade 0 to Grade 4)
- **Image Size:** 256×256 pixels
- **Format:** JPEG

| Class | Images |
|-------|--------|
| Healthy (Grade 0) | 1000 |
| Mild DR (Grade 1) | 370 |
| Moderate DR (Grade 2) | 900 |
| Proliferative DR (Grade 3) | 290 |
| Severe DR (Grade 4) | 190 |

---

## 🖥️ How to Use the Web App
1. Open the live demo link
2. Upload a retinal fundus image (JPG/PNG)
3. Click **"Analyze Image"**
4. View the predicted DR grade and confidence score
5. Read the medical advice for the detected grade

---

## 📈 Training Details
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-4 → 1e-5 |
| Batch Size | 16 |
| Phase 1 Epochs | 5 (frozen backbone) |
| Phase 2 Epochs | 20 (fine-tuning) |
| Loss Function | Categorical Crossentropy |
| Input Size | 224×224×3 |

---

## 🔮 Future Work
- [ ] GradCAM visualization for lesion highlighting
- [ ] Mobile deployment with MobileNet
- [ ] Flask/FastAPI REST API
- [ ] Federated learning for privacy
- [ ] Larger dataset training on GPU

---

## 👨‍💻 About
**Minor Project Submission**
B.Tech Computer Science Engineering

---

## 📄 License
This project is licensed under the MIT License.
