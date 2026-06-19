# -*- coding: utf-8 -*-
"""app.py — Face PCA Studio

UI/UX didesain ulang mengikuti 6 mockup (tema navy/indigo gelap, sidebar
ikon ramping, kartu upload berlabel vertikal, kartu metrik, dsb) dan
dianimasikan menggunakan animate.css (https://unpkg.com/animate.css@4.1.1).
"""

import os
import io

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Face PCA Studio",
    page_icon="🪞",
    layout="wide",
    initial_sidebar_state="expanded",
)

ANIMATE_CSS_URL = "https://unpkg.com/animate.css@4.1.1/animate.css"

# =========================================================
# THEME PALETTES (dark = sesuai mockup, light = mode alternatif)
# =========================================================
THEMES = {
    "dark": {
        "bg_outer": "radial-gradient(circle at 25% 15%, #3d42a0 0%, #262b66 35%, "
                    "#121432 78%)",
        "panel_bg": "linear-gradient(160deg, #1b1f47 0%, #0f1129 100%)",
        "card_bg": "linear-gradient(160deg, #1d2249 0%, #11132c 100%)",
        "card_border": "#2b3066",
        "sidebar_bg": "linear-gradient(180deg, #05060f 0%, #171b3e 55%, #2d3175 100%)",
        "strip_bg": "linear-gradient(180deg, #262c63 0%, #10122a 100%)",
        "text_primary": "#f5f6ff",
        "text_secondary": "#a6abdb",
        "accent": "#5eead4",
        "accent_strong": "#8b93ff",
        "shadow": "0 18px 42px rgba(4,5,18,0.45)",
    },
    "light": {
        "bg_outer": "linear-gradient(135deg, #eef0ff 0%, #e2e5ff 100%)",
        "panel_bg": "#ffffff",
        "card_bg": "#f6f7ff",
        "card_border": "#dcdcf5",
        "sidebar_bg": "linear-gradient(180deg, #ffffff 0%, #e7e9ff 100%)",
        "strip_bg": "linear-gradient(180deg, #dde0ff 0%, #c6caf7 100%)",
        "text_primary": "#1c1b3a",
        "text_secondary": "#6b6890",
        "accent": "#0d9488",
        "accent_strong": "#4f46e5",
        "shadow": "0 14px 32px rgba(30,27,75,0.12)",
    },
}

UPLOAD_CARD_LABELS = {
    "card_foto1": "FOTO PERTAMA",
    "card_foto2": "FOTO KEDUA",
    "card_original": "FOTO ASLI",
    "card_result": "HASIL KOMPRESI",
}

# =========================================================
# CSS TEMPLATE (token-based, di-replace per tema agar aman dari
# konflik kurung kurawal f-string)
# =========================================================
BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');
@import url('__ANIMATE_URL__');

html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

.stApp { background: __BG_OUTER__; }

/* ---------- panel utama ---------- */
.main .block-container {
    background: __PANEL_BG__;
    border-radius: 28px;
    padding: 1.9rem 2.3rem 2.6rem 2.3rem;
    margin: 14px 14px 14px 0;
    box-shadow: __SHADOW__;
    animation: fadeIn 0.7s ease both;
}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
    background: __SIDEBAR_BG__;
    width: 92px !important;
    min-width: 92px !important;
    border-radius: 0 26px 26px 0;
    margin: 14px 0 14px 14px;
    box-shadow: __SHADOW__;
    animation: fadeInLeft 0.6s ease both;
}
section[data-testid="stSidebar"] > div {
    padding: 1.1rem 0 1.2rem 0 !important;
    height: 100%;
}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.6rem;
}
.sidebar-logo {
    font-size: 1.9rem;
    text-align: center;
    margin-bottom: 0.6rem;
    animation: zoomIn 0.7s ease 0.1s both;
}
.sidebar-spacer { flex: 1 1 auto; }

section[data-testid="stSidebar"] div[role="radiogroup"] {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: transparent;
    border: none !important;
    margin: 0 !important;
    padding: 0 !important;
    width: 50px;
    height: 50px;
    border-radius: 14px;
    display: flex !important;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: background 0.2s ease, transform 0.2s ease;
    animation: fadeInLeft 0.5s ease both;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(1) { animation-delay: 0.05s; }
section[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(2) { animation-delay: 0.10s; }
section[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(3) { animation-delay: 0.15s; }
section[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(4) { animation-delay: 0.20s; }
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.10);
    transform: scale(1.08);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: rgba(255,255,255,0.16);
    box-shadow: inset 3px 0 0 __ACCENT__;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { display: none; }
section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    font-size: 1.45rem;
    margin: 0;
    color: __TEXT_PRIMARY__ !important;
}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] { display: none; }

section[data-testid="stSidebar"] button {
    width: 46px;
    height: 46px;
    border-radius: 50% !important;
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: __TEXT_PRIMARY__ !important;
    font-size: 1.2rem !important;
    box-shadow: none !important;
    animation: zoomIn 0.6s ease 0.25s both;
}
section[data-testid="stSidebar"] button:hover {
    animation: pulse 0.5s ease;
    background: rgba(255,255,255,0.16) !important;
}

/* ---------- kartu umum (st.container border=True) ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    background: __CARD_BG__ !important;
    border: 1px solid __CARD_BORDER__ !important;
    border-radius: 20px !important;
    padding: 1.3rem 1.5rem !important;
    box-shadow: 0 10px 26px rgba(0,0,0,0.22);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    animation: fadeInUp 0.55s ease both;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div:hover {
    transform: translateY(-3px);
    box-shadow: 0 16px 34px rgba(0,0,0,0.30);
}
div[data-testid="stVerticalBlockBorderWrapper"] h1,
div[data-testid="stVerticalBlockBorderWrapper"] h2,
div[data-testid="stVerticalBlockBorderWrapper"] h3,
div[data-testid="stVerticalBlockBorderWrapper"] p,
div[data-testid="stVerticalBlockBorderWrapper"] label,
div[data-testid="stVerticalBlockBorderWrapper"] span {
    color: __TEXT_PRIMARY__ !important;
}

/* ---------- judul & elemen teks kustom ---------- */
.page-title {
    font-weight: 800;
    font-size: 2.3rem;
    letter-spacing: 0.5px;
    color: __TEXT_PRIMARY__;
    margin: 0.4rem 0 1.1rem 0;
}
.about-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(94,234,212,0.14);
    color: __ACCENT__;
    font-weight: 700;
    font-size: 0.78rem;
    letter-spacing: 1.5px;
    padding: 5px 16px;
    border-radius: 999px;
    margin-bottom: 0.7rem;
}
.info-dot {
    width: 16px; height: 16px;
    border-radius: 50%;
    background: __ACCENT__;
    color: #06231f;
    font-style: italic;
    font-weight: 800;
    font-size: 0.7rem;
    display: inline-flex;
    align-items: center;
    justify-content: center;
}
.section-card-title {
    font-weight: 700;
    font-size: 1.15rem;
    color: __TEXT_PRIMARY__;
    margin-bottom: 0.6rem;
}
.muted-caption { color: __TEXT_SECONDARY__; font-size: 0.86rem; }

.info-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    background: __CARD_BG__;
    border: 1px solid __CARD_BORDER__;
    border-radius: 16px;
    padding: 0.95rem 1.4rem;
    color: __TEXT_PRIMARY__;
    font-weight: 500;
    margin-bottom: 1.3rem;
    animation: fadeIn 0.6s ease both;
}
.info-bar .info-dot { width: 20px; height: 20px; font-size: 0.85rem; }

.decor-icon-card {
    height: 100%;
    min-height: 230px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 4.6rem;
    animation: zoomIn 0.8s ease 0.15s both;
}
.tool-icon-box {
    height: 100%;
    min-height: 96px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2.2rem;
    animation: zoomIn 0.6s ease both;
}
.feature-icon {
    font-size: 1.4rem;
    margin-right: 8px;
}

/* ---------- metric cards ---------- */
div[data-testid="stMetric"] {
    background: __CARD_BG__ !important;
    border: 1px solid __CARD_BORDER__ !important;
    border-radius: 18px !important;
    padding: 1.0rem 1.2rem !important;
    box-shadow: 0 8px 20px rgba(0,0,0,0.20);
    animation: zoomIn 0.5s ease both;
}
div[data-testid="stMetricLabel"] { color: __TEXT_SECONDARY__ !important; font-weight: 600; }
div[data-testid="stMetricValue"] { color: __TEXT_PRIMARY__ !important; font-weight: 700; }

/* ---------- tombol area utama ---------- */
.main .stButton > button {
    border-radius: 12px;
    font-weight: 600;
    background: linear-gradient(120deg, __ACCENT_STRONG__, __ACCENT__);
    color: #0b0d22;
    border: none;
    padding: 0.65rem 1.5rem;
    box-shadow: 0 8px 20px rgba(0,0,0,0.25);
    transition: transform 0.2s ease;
}
.main .stButton > button:hover {
    animation: pulse 0.5s ease;
    transform: scale(1.02);
}

/* ---------- uploader ---------- */
div[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: 1px dashed __CARD_BORDER__ !important;
    border-radius: 16px !important;
    min-height: 280px;
    display: flex !important;
    align-items: center;
    justify-content: center;
}
div[data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }
div[data-testid="stFileUploaderDropzone"] button {
    font-size: 0 !important;
    background: __CARD_BG__ !important;
    border: 1px solid __CARD_BORDER__ !important;
    color: __TEXT_PRIMARY__ !important;
    border-radius: 12px !important;
    padding: 0.7rem 2.2rem !important;
    box-shadow: 0 6px 16px rgba(0,0,0,0.25);
    animation: fadeIn 0.7s ease both;
}
div[data-testid="stFileUploaderDropzone"] button::after {
    content: "Upload";
    font-size: 0.95rem;
    font-weight: 600;
}
div[data-testid="stFileUploaderDropzone"] button:hover { animation: pulse 0.5s ease; }
div[data-testid="stFileUploaderDropzone"] svg { display: none; }

/* ---------- radio horizontal (grayscale/warna) ---------- */
div[data-testid="stRadio"] div[role="radiogroup"] {
    background: __CARD_BG__;
    border: 1px solid __CARD_BORDER__;
    border-radius: 999px;
    padding: 6px 18px;
    display: inline-flex;
    gap: 22px;
}
.main div[data-testid="stRadio"] label p { color: __TEXT_PRIMARY__ !important; font-weight: 600; }

/* ---------- slider ---------- */
div[data-testid="stSlider"] [role="slider"] {
    background: #ffffff !important;
    box-shadow: 0 0 0 4px rgba(94,234,212,0.25) !important;
}
div[data-baseweb="slider"] > div > div:nth-child(2) {
    background: __ACCENT__ !important;
}
.main label p { color: __TEXT_PRIMARY__ !important; font-weight: 600; }
.main .stCaption, .main [data-testid="stCaptionContainer"] { color: __TEXT_SECONDARY__ !important; }

/* ---------- gambar di dalam kartu ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] img {
    border-radius: 14px;
    animation: zoomIn 0.5s ease both;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}
"""

UPLOAD_CARD_CSS = """
.st-key-__KEY__ {
    position: relative !important;
    overflow: hidden !important;
    padding-right: 60px !important;
    min-height: 360px;
    animation: fadeInUp 0.6s ease both;
}
.st-key-__KEY__::after {
    content: "🪞  __LABEL__";
    position: absolute;
    top: 0; right: 0; bottom: 0;
    width: 54px;
    background: __STRIP_BG__;
    color: __TEXT_PRIMARY__;
    writing-mode: vertical-rl;
    text-orientation: mixed;
    display: flex;
    align-items: center;
    justify-content: center;
    letter-spacing: 3px;
    font-weight: 700;
    font-size: 0.82rem;
    border-left: 1px solid __CARD_BORDER__;
    animation: fadeInRight 0.6s ease 0.1s both;
}
"""


def inject_css(theme_name: str):
    t = THEMES[theme_name]
    css = BASE_CSS.replace("__ANIMATE_URL__", ANIMATE_CSS_URL)
    for key, val in t.items():
        css = css.replace("__" + key.upper() + "__", val)

    for key, label in UPLOAD_CARD_LABELS.items():
        block = UPLOAD_CARD_CSS.replace("__KEY__", key).replace("__LABEL__", label)
        for tkey, tval in t.items():
            block = block.replace("__" + tkey.upper() + "__", tval)
        css += block

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# =========================================================
# SESSION STATE — tema
# =========================================================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

inject_css(st.session_state.theme)
THEME = THEMES[st.session_state.theme]

# =========================================================
# LOAD MODEL
# =========================================================
MODEL_CANDIDATES = [
    "model/model_with_me.pkl",
    "model_with_me.pkl",
]


@st.cache_resource
def load_model():
    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    return None


model = load_model()

if model is None:
    st.error(
        "⚠️ File `model_with_me.pkl` tidak ditemukan. Pastikan file berada di "
        "root project atau di folder `model/`."
    )
    st.stop()

pca = model.get("pca")
X_train_pca = model.get("X_train_pca")
y_train = model.get("y_train")
X_test_pca = model.get("X_test_pca")
y_test = model.get("y_test")
labels_all = model.get("labels")


# =========================================================
# HELPERS
# =========================================================
def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    img_arr = np.array(img)
    img_eq = cv2.equalizeHist(img_arr)
    img_blur = cv2.GaussianBlur(img_eq, (3, 3), 0)
    img_resized = cv2.resize(img_blur, (100, 100))
    return img_resized.flatten() / 255.0


def svd_compress_gray(img_arr, k):
    U, S, Vt = np.linalg.svd(img_arr, full_matrices=False)
    k = min(k, len(S))
    reconstructed = (U[:, :k] * S[:k]) @ Vt[:k, :]
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    return reconstructed, S


def svd_compress_color(img_arr, k):
    channels = []
    s_values = None
    for c in range(3):
        ch = img_arr[:, :, c].astype(float)
        U, S, Vt = np.linalg.svd(ch, full_matrices=False)
        k_ = min(k, len(S))
        rec = (U[:, :k_] * S[:k_]) @ Vt[:k_, :]
        channels.append(np.clip(rec, 0, 255))
        if s_values is None:
            s_values = S
    reconstructed = np.stack(channels, axis=2).astype(np.uint8)
    return reconstructed, s_values


def style_fig(fig, ax, theme):
    """Samakan warna chart matplotlib dengan tema aktif & buat transparan."""
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_color(theme["card_border"])
    ax.tick_params(colors=theme["text_secondary"])
    ax.xaxis.label.set_color(theme["text_secondary"])
    ax.yaxis.label.set_color(theme["text_secondary"])
    ax.title.set_color(theme["text_primary"])
    ax.grid(alpha=0.15, color=theme["text_secondary"])
    legend = ax.get_legend()
    if legend is not None:
        legend.get_frame().set_alpha(0)
        for text in legend.get_texts():
            text.set_color(theme["text_primary"])


# =========================================================
# SIDEBAR — NAVIGASI IKON
# =========================================================
NAV_ITEMS = [
    ("🏠", "Home"),
    ("📊", "EDA & Visualisasi"),
    ("🔍", "Bandingkan Dua Wajah"),
    ("🗜️", "Kompresi Gambar (SVD)"),
]
NAV_ICONS = [i for i, _ in NAV_ITEMS]
NAV_LABELS = [l for _, l in NAV_ITEMS]

with st.sidebar:
    st.markdown('<div class="sidebar-logo">🪞</div>', unsafe_allow_html=True)
    chosen_icon = st.radio(
        "Navigasi",
        NAV_ICONS,
        label_visibility="collapsed",
        key="nav_icon",
    )
    page = NAV_LABELS[NAV_ICONS.index(chosen_icon)]

    st.markdown('<div class="sidebar-spacer"></div>', unsafe_allow_html=True)

    theme_icon = "🌙" if st.session_state.theme == "dark" else "☀️"
    if st.button(theme_icon, key="theme_btn"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()


# =========================================================
# PAGE: HOME
# =========================================================
def render_home():
    top_left, top_right = st.columns([2, 1])
    with top_left:
        with st.container(border=True):
            st.markdown(
                '<span class="about-badge"><span class="info-dot">i</span> ABOUT</span>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="page-title">FACE PCA STUDIO</div>', unsafe_allow_html=True)
            st.write(
                "Eksplorasi wajah manusia lewat Principal Component Analysis (PCA) "
                "dan Singular Value Decomposition (SVD) — dari eigenfaces sampai "
                "kompresi gambar, semuanya interaktif!"
            )
    with top_right:
        with st.container(border=True):
            st.markdown('<div class="decor-icon-card">🪞</div>', unsafe_allow_html=True)

    st.write("")

    left_col, right_col = st.columns([1, 1.5])

    with left_col:
        features = [
            ("📊", "EDA & Visualisasi",
             "Lihat distribusi data, mean face, eigenfaces, dan seberapa banyak "
             "variance yang ditangkap oleh PCA."),
            ("🔍", "Bandingkan 2 Wajah",
             "Upload dua foto dan cek apakah itu orang yang sama lewat cosine "
             "similarity di ruang PCA."),
            ("🗜️", "Kompresi Gambar",
             "Upload gambar apa saja dan lihat bagaimana SVD bisa mengompresnya "
             "dengan berbagai tingkat kualitas."),
        ]
        for icon, title, desc in features:
            with st.container(border=True):
                st.markdown(
                    f'<div class="section-card-title">'
                    f'<span class="feature-icon">{icon}</span>{title}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f'<span class="muted-caption">{desc}</span>', unsafe_allow_html=True)

    with right_col:
        n_components = pca.n_components_ if pca is not None else "-"
        n_identities = len(np.unique(y_train)) if y_train is not None else "-"
        n_photos = len(y_train) if y_train is not None else "-"

        m0, m1, m2, m3 = st.columns([0.8, 1, 1, 1])
        with m0:
            with st.container(border=True):
                st.markdown('<div class="tool-icon-box">🛠️</div>', unsafe_allow_html=True)
        with m1:
            st.metric("Komponen PCA", n_components)
        with m2:
            st.metric("Jumlah Identitas", n_identities)
        with m3:
            st.metric("Total Foto Training", n_photos)

        with st.container(border=True):
            st.markdown('<div class="section-card-title">Cara Kerja Singkat</div>', unsafe_allow_html=True)
            st.write(
                "Setiap foto wajah diubah jadi grayscale, disamakan pencahayaannya, "
                "diubah ukurannya jadi 100×100 piksel, lalu diproyeksikan ke ruang "
                "PCA hasil training. Kemiripan dua wajah diukur lewat cosine "
                "similarity antar vektor PCA-nya — semakin mendekati 1, semakin "
                "mirip kedua wajah tersebut."
            )


# =========================================================
# PAGE: EDA & VISUALISASI
# =========================================================
def render_eda():
    if y_train is None or pca is None:
        st.warning("Data model tidak lengkap untuk menampilkan EDA.")
        return

    col_eigen, col_mid, col_right = st.columns([1, 1.7, 1.2])

    with col_eigen:
        st.markdown('<div class="section-card-title">Top Eigenfaces</div>', unsafe_allow_html=True)
        n_show = min(10, pca.components_.shape[0])
        for row_start in range(0, n_show, 2):
            cols = st.columns(2)
            for offset, col in enumerate(cols):
                idx = row_start + offset
                if idx >= n_show:
                    continue
                with col:
                    with st.container(border=True):
                        fig, ax = plt.subplots(figsize=(2, 2))
                        ax.imshow(pca.components_[idx].reshape(100, 100), cmap="gray")
                        ax.axis("off")
                        fig.patch.set_alpha(0.0)
                        st.pyplot(fig, use_container_width=True)

    with col_mid:
        with st.container(border=True):
            st.markdown('<div class="section-card-title">Distribusi Foto per Identitas</div>', unsafe_allow_html=True)
            counts = pd.Series(y_train).value_counts().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(6, 3.4))
            ax.bar(range(len(counts)), counts.values, color=THEME["accent_strong"])
            ax.set_xlabel("Identitas")
            ax.set_ylabel("Jumlah Foto")
            style_fig(fig, ax, THEME)
            st.pyplot(fig, use_container_width=True)
            st.markdown(
                f'<span class="muted-caption">Rata-rata {counts.mean():.1f} foto per '
                f'orang dari {len(counts)} identitas.</span>',
                unsafe_allow_html=True,
            )

        if X_test_pca is not None and y_test is not None and X_train_pca is not None:
            with st.container(border=True):
                st.markdown('<div class="section-card-title">Akurasi Model (Pengenalan Wajah)</div>', unsafe_allow_html=True)
                sims_matrix = cosine_similarity(X_test_pca, X_train_pca)
                best_idx = np.argmax(sims_matrix, axis=1)
                preds = np.array(y_train)[best_idx]
                acc = float(np.mean(preds == np.array(y_test)))
                st.metric("Akurasi pada data test", f"{acc * 100:.1f}%")
                st.markdown(
                    f'<span class="muted-caption">Diuji pada {len(y_test)} foto data test.</span>',
                    unsafe_allow_html=True,
                )

    with col_right:
        with st.container(border=True):
            st.markdown('<div class="section-card-title">Mean Faces</div>', unsafe_allow_html=True)
            mean_face = pca.mean_.reshape(100, 100)
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(mean_face, cmap="gray")
            ax.axis("off")
            fig.patch.set_alpha(0.0)
            st.pyplot(fig, use_container_width=True)
            st.markdown(
                '<span class="muted-caption">Wajah rata-rata dari seluruh data training.</span>',
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            st.markdown('<div class="section-card-title">Explained Variance</div>', unsafe_allow_html=True)
            cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
            fig, ax = plt.subplots(figsize=(4, 3.2))
            ax.plot(cumvar, color=THEME["accent_strong"], linewidth=2.5)
            ax.fill_between(range(len(cumvar)), cumvar, color=THEME["accent_strong"], alpha=0.15)
            ax.axhline(95, color=THEME["accent"], linestyle="--", label="95%")
            ax.set_xlabel("Jumlah Komponen")
            ax.set_ylabel("Variance (%)")
            ax.legend()
            style_fig(fig, ax, THEME)
            st.pyplot(fig, use_container_width=True)
            st.markdown(
                f'<span class="muted-caption">Total variance dipertahankan: '
                f'{cumvar[-1]:.1f}% dengan {pca.n_components_} komponen.</span>',
                unsafe_allow_html=True,
            )


# =========================================================
# PAGE: BANDINGKAN DUA WAJAH
# =========================================================
def render_compare():
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True, key="card_foto1"):
            foto_pertama = st.file_uploader(
                "Foto Pertama", type=["jpg", "jpeg", "png"], key="foto1",
                label_visibility="collapsed",
            )
            if foto_pertama:
                st.image(foto_pertama, use_container_width=True)
    with col2:
        with st.container(border=True, key="card_foto2"):
            foto_kedua = st.file_uploader(
                "Foto Kedua", type=["jpg", "jpeg", "png"], key="foto2",
                label_visibility="collapsed",
            )
            if foto_kedua:
                st.image(foto_kedua, use_container_width=True)

    st.write("")
    threshold = st.slider("Ambang Batas Kemiripan (Threshold)", 0.0, 1.0, 0.50, 0.01)

    if foto_pertama and foto_kedua:
        if st.button("Bandingkan Sekarang"):
            face1 = preprocess(foto_pertama.getvalue())
            face2 = preprocess(foto_kedua.getvalue())

            z1 = pca.transform(face1.reshape(1, -1))
            z2 = pca.transform(face2.reshape(1, -1))
            sim = float(cosine_similarity(z1, z2)[0][0])

            with st.container(border=True):
                st.markdown('<div class="section-card-title">Hasil</div>', unsafe_allow_html=True)
                st.progress(min(max(sim, 0.0), 1.0))
                st.metric("Cosine Similarity", f"{sim:.4f}")
                if sim >= threshold:
                    st.success("✅ SAMA ORANG")
                else:
                    st.error("❌ BEDA ORANG")


# =========================================================
# PAGE: KOMPRESI GAMBAR (SVD)
# =========================================================
def render_svd():
    uploaded = st.file_uploader(
        "Upload gambar",
        type=["jpg", "jpeg", "png"],
        key="svd_img_uploader",
        label_visibility="collapsed",
    )

    if not uploaded:
        st.markdown(
            '<div class="info-bar"><span class="info-dot">i</span> '
            'Upload gambar untuk mulai bereksperimen dengan kompresi SVD</div>',
            unsafe_allow_html=True,
        )
        return

    img = Image.open(uploaded)
    mode = st.radio("Mode warna", ["Grayscale", "Warna (RGB)"], horizontal=True, label_visibility="collapsed")
    img_arr = (
        np.array(img.convert("L"))
        if mode == "Grayscale"
        else np.array(img.convert("RGB"))
    )

    h, w = img_arr.shape[:2]
    max_k = min(h, w, 300)
    default_k = min(50, max_k)
    k = st.slider("Jumlah komponen (k)", 1, max_k, default_k)

    if mode == "Grayscale":
        reconstructed, singular_values = svd_compress_gray(img_arr.astype(float), k)
    else:
        reconstructed, singular_values = svd_compress_color(img_arr.astype(float), k)

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True, key="card_original"):
            st.image(img_arr, use_container_width=True, clamp=True)
    with col2:
        with st.container(border=True, key="card_result"):
            st.image(reconstructed, use_container_width=True, clamp=True)

    n_channels = 1 if mode == "Grayscale" else 3
    original_size = h * w * n_channels
    compressed_size = k * (h + w + 1) * n_channels
    ratio = original_size / compressed_size

    m1, m2, m3 = st.columns(3)
    m1.metric("Ukuran Asli (nilai)", f"{original_size:,}")
    m2.metric("Ukuran Terkompresi (nilai)", f"{compressed_size:,}")
    m3.metric("Rasio Kompresi (nilai)", f"{ratio:.1f}x")

    with st.container(border=True):
        st.markdown('<div class="section-card-title">Singular Value Decay</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(singular_values, color=THEME["accent_strong"], linewidth=2)
        ax.axvline(k, color=THEME["accent"], linestyle="--", label=f"k={k}")
        ax.set_yscale("log")
        ax.set_xlabel("Index")
        ax.set_ylabel("Nilai Singular (skala log)")
        ax.legend()
        style_fig(fig, ax, THEME)
        st.pyplot(fig, use_container_width=True)
        st.markdown(
            '<span class="muted-caption">Sebagian besar \'informasi\' gambar '
            'terkonsentrasi di singular value pertama — itulah mengapa gambar '
            'masih cukup jelas walau k kecil.</span>',
            unsafe_allow_html=True,
        )


# =========================================================
# ROUTER
# =========================================================
if page == "Home":
    render_home()
elif page == "EDA & Visualisasi":
    render_eda()
elif page == "Bandingkan Dua Wajah":
    render_compare()
elif page == "Kompresi Gambar (SVD)":
    render_svd()
