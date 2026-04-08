import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Cotton Leaf Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# ---------------------- LOAD MODEL ----------------------
model = load_model("model.h5")

class_names = [
    'Alteneria',
    'Aphids',
    'Bacterial blight',
    'Curl virus',
    'Fusarium Wilt',
    'Healthy',
    'Herbicide Growth Damage',
    'Leaf Hopper Jassids',
    'Leaf Redding',
    'Leaf Variegation',
    'Mildew',
    'Verticillium Wilt',
    'target spot'
]

disease_info = {
    'Alteneria': "Alternaria is a fungal disease that causes brown lesions and damaged spots on cotton leaves.",
    'Aphids': "Aphids are sap-sucking pests that weaken the plant and may spread other infections.",
    'Bacterial blight': "Bacterial blight causes angular leaf spots and reduces plant health and yield.",
    'Curl virus': "Curl virus causes curling, distortion, and poor leaf development in cotton plants.",
    'Fusarium Wilt': "Fusarium wilt is a fungal disease that causes yellowing, wilting, and plant decline.",
    'Healthy': "The cotton leaf appears healthy and free from major disease symptoms.",
    'Herbicide Growth Damage': "This indicates damage due to herbicide exposure, affecting normal plant growth.",
    'Leaf Hopper Jassids': "Jassid infestation leads to yellowing, edge burning, and curling symptoms.",
    'Leaf Redding': "Leaf reddening causes red or reddish-purple discoloration in the leaves.",
    'Leaf Variegation': "Leaf variegation shows irregular color variations on the leaf surface.",
    'Mildew': "Mildew is a fungal disease causing white or grayish patches and leaf damage.",
    'Verticillium Wilt': "Verticillium wilt affects the vascular system, leading to wilting and chlorosis.",
    'target spot': "Target spot causes circular lesions and may spread across the leaf surface."
}

IMG_SIZE = 224

# ---------------------- CUSTOM STYLE ----------------------
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #1b5e20;
        margin-bottom: 5px;
    }
    .sub-title {
        text-align: center;
        font-size: 20px;
        color: #4e5d6c;
        margin-bottom: 30px;
    }
    .section-title {
        font-size: 24px;
        font-weight: bold;
        color: #2e7d32;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .result-box {
        padding: 18px;
        border-radius: 12px;
        background-color: #f1f8e9;
        border: 1px solid #c5e1a5;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .info-box {
        padding: 16px;
        border-radius: 12px;
        background-color: #f9f9f9;
        border: 1px solid #dddddd;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------- SIDEBAR ----------------------
st.sidebar.title("Project Menu")
st.sidebar.markdown("### Cotton Leaf Disease Detection")
st.sidebar.write("This application predicts cotton leaf diseases using a trained MobileNetV2 deep learning model.")

st.sidebar.markdown("### Project Details")
st.sidebar.write("**Model:** MobileNetV2")
st.sidebar.write("**Input Size:** 224 × 224")
st.sidebar.write("**Total Classes:** 13")
st.sidebar.write("**Platform:** Streamlit")

st.sidebar.markdown("### Instructions")
st.sidebar.write("1. Upload a cotton leaf image.")
st.sidebar.write("2. Click on the Predict button.")
st.sidebar.write("3. View predicted disease, confidence, and class probabilities.")

# ---------------------- HOME PAGE HEADER ----------------------
st.markdown('<div class="main-title">🌿 Cotton Leaf Disease Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">A Deep Learning Based Prediction Interface using MobileNetV2</div>', unsafe_allow_html=True)

# ---------------------- PROJECT DESCRIPTION ----------------------
st.markdown('<div class="section-title">Project Overview</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
This application is developed for automatic cotton leaf disease classification using a trained MobileNetV2 model.
The user can upload a cotton leaf image, and the system predicts the most likely disease category along with confidence score,
disease description, and class-wise probabilities.
</div>
""", unsafe_allow_html=True)

# ---------------------- IMAGE PREPROCESS ----------------------
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# ---------------------- UPLOAD SECTION ----------------------
st.markdown('<div class="section-title">Upload Leaf Image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a cotton leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Cotton Leaf Image")

    with col2:
        st.markdown("### Prediction Panel")
        if st.button("Predict Disease"):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)

            pred_index = int(np.argmax(prediction))
            predicted_class = class_names[pred_index]
            confidence = float(np.max(prediction)) * 100

            st.markdown(f"""
            <div class="result-box">
                <h3>Prediction Result</h3>
                <p><b>Predicted Disease:</b> {predicted_class}</p>
                <p><b>Confidence Score:</b> {confidence:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

            if confidence >= 90:
                st.success("The model is highly confident about this prediction.")
            elif confidence >= 75:
                st.warning("The model is moderately confident about this prediction.")
            else:
                st.error("Low-confidence prediction. Please verify with additional images or expert opinion.")

            st.markdown("### Disease Description")
            st.write(disease_info.get(predicted_class, "No description available."))

            prob_df = pd.DataFrame({
                "Disease Class": class_names,
                "Probability (%)": [float(p * 100) for p in prediction[0]]
            }).sort_values(by="Probability (%)", ascending=False)

            st.markdown("### Prediction Probabilities")
            st.dataframe(prob_df, use_container_width=True)

            st.markdown("### Probability Chart")
            st.bar_chart(prob_df.set_index("Disease Class"))

# ---------------------- FOOTER ----------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    Developed for Cotton Leaf Disease Prediction using  Deep Learning
    </div>
    """,
    unsafe_allow_html=True
)