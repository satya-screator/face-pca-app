import streamlit as st
import pickle
import numpy as np
import cv2
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import io

# Load model
@st.cache_resource
def load_model():
    with open('model_with_me.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()
pca         = model['pca']
X_train_pca = model['X_train_pca']
y_train     = model['y_train']

# Preprocessing
def preprocess(img_data):
    img = Image.open(io.BytesIO(img_data)).convert('L')
    img_arr = np.array(img)
    img_eq = cv2.equalizeHist(img_arr)
    img_blur = cv2.GaussianBlur(img_eq, (3, 3), 0)
    img_resized = cv2.resize(img_blur, (100, 100))
    return img_resized.flatten() / 255.0

# UI
st.title("👤 Face Similarity PCA")
st.write("Upload foto masa kecil dan foto sekarang untuk dibandingkan")

col1, col2 = st.columns(2)

with col1:
    foto_kecil = st.file_uploader("📸 Foto Masa Kecil", type=['jpg','jpeg','png'])
    if foto_kecil:
        st.image(foto_kecil, caption="Masa Kecil", use_container_width=True)

with col2:
    foto_sekarang = st.file_uploader("📸 Foto Sekarang", type=['jpg','jpeg','png'])
    if foto_sekarang:
        st.image(foto_sekarang, caption="Sekarang", use_container_width=True)

if foto_kecil and foto_sekarang:
    if st.button("🔍 Bandingkan"):
        face1 = preprocess(foto_kecil.read())
        face2 = preprocess(foto_sekarang.read())

        z1 = pca.transform(face1.reshape(1, -1))
        z2 = pca.transform(face2.reshape(1, -1))
        sim = cosine_similarity(z1, z2)[0][0]

        st.divider()
        st.metric("Cosine Similarity", f"{sim:.4f}")

        if sim >= 0.50:
            st.success("✅ SAMA ORANG")
        else:
            st.error("❌ BEDA ORANG")
