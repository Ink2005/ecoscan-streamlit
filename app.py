import os
from pathlib import Path

# Keras backend harus diset sebelum import keras.
os.environ.setdefault("KERAS_BACKEND", "openvino")

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model" / "waste_classifier_final.keras"
CONFUSION_PATH = APP_DIR / "assets" / "confusion_matrix.png"
IMG_SIZE = (224, 224)

# Mapping dari notebook:
# O = 0 = Organic / Organik
# R = 1 = Inorganic / Anorganik

st.set_page_config(
    page_title="EcoScan - Klasifikasi Sampah",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 12% 10%, rgba(34, 197, 94, 0.20), transparent 34%),
            radial-gradient(circle at 88% 18%, rgba(20, 184, 166, 0.16), transparent 28%),
            radial-gradient(circle at 55% 85%, rgba(59, 130, 246, 0.11), transparent 32%),
            linear-gradient(135deg, #07110d 0%, #0b1117 45%, #0d1720 100%);
        color: #f8fafc;
    }

    section[data-testid="stSidebar"] {
        background: rgba(7, 17, 13, 0.82);
        border-right: 1px solid rgba(148, 163, 184, 0.16);
        backdrop-filter: blur(18px);
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    .block-container {
        padding-top: 2.0rem;
        padding-bottom: 2.5rem;
        max-width: 1240px;
    }

    .hero {
        position: relative;
        padding: 2.2rem 2.2rem;
        border-radius: 28px;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.86), rgba(6, 78, 59, 0.56));
        border: 1px solid rgba(148, 163, 184, 0.20);
        box-shadow: 0 24px 80px rgba(0, 0, 0, 0.28);
        overflow: hidden;
        margin-bottom: 1.3rem;
    }

    .hero::after {
        content: "";
        position: absolute;
        right: -90px;
        top: -90px;
        width: 260px;
        height: 260px;
        border-radius: 999px;
        background: rgba(34, 197, 94, 0.22);
        filter: blur(6px);
    }

    .eyebrow {
        display: inline-flex;
        gap: 0.55rem;
        align-items: center;
        padding: 0.45rem 0.75rem;
        border-radius: 999px;
        color: #bbf7d0;
        background: rgba(34, 197, 94, 0.13);
        border: 1px solid rgba(74, 222, 128, 0.24);
        font-size: 0.84rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin-bottom: 1.0rem;
    }

    .hero h1 {
        color: #ffffff;
        font-size: clamp(2.05rem, 4vw, 4.05rem);
        line-height: 1.02;
        letter-spacing: -0.06em;
        margin: 0 0 0.8rem 0;
        font-weight: 900;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 1.05rem;
        line-height: 1.75;
        max-width: 760px;
        margin: 0;
    }

    .hero-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.85rem;
        margin-top: 1.4rem;
        max-width: 760px;
    }

    .mini-stat {
        padding: 1rem;
        border-radius: 18px;
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.16);
    }

    .mini-stat .num {
        font-size: 1.15rem;
        color: #ffffff;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }

    .mini-stat .label {
        color: #94a3b8;
        font-size: 0.84rem;
        line-height: 1.35;
    }

    .glass-card {
        padding: 1.25rem;
        border-radius: 24px;
        background: rgba(15, 23, 42, 0.62);
        border: 1px solid rgba(148, 163, 184, 0.16);
        box-shadow: 0 16px 50px rgba(0, 0, 0, 0.20);
        backdrop-filter: blur(16px);
        margin-bottom: 1rem;
    }

    .soft-card {
        padding: 1.15rem;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(148, 163, 184, 0.14);
        margin-bottom: 0.85rem;
    }

    .section-title {
        color: #f8fafc;
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.3rem;
    }

    .section-desc {
        color: #94a3b8;
        font-size: 0.92rem;
        line-height: 1.55;
        margin-bottom: 1rem;
    }

    .result-organic {
        padding: 1.3rem;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.22), rgba(6, 95, 70, 0.18));
        border: 1px solid rgba(74, 222, 128, 0.35);
        box-shadow: 0 16px 50px rgba(16, 185, 129, 0.12);
    }

    .result-inorganic {
        padding: 1.3rem;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.20), rgba(234, 88, 12, 0.16));
        border: 1px solid rgba(251, 191, 36, 0.35);
        box-shadow: 0 16px 50px rgba(245, 158, 11, 0.11);
    }

    .result-label {
        color: #ffffff;
        font-size: 2.05rem;
        font-weight: 900;
        letter-spacing: -0.05em;
        margin-bottom: 0.15rem;
    }

    .result-sub {
        color: #d1fae5;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }

    .score-row {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.75rem;
        margin-top: 0.9rem;
    }

    .score-box {
        padding: 0.9rem;
        border-radius: 16px;
        background: rgba(15, 23, 42, 0.42);
        border: 1px solid rgba(255, 255, 255, 0.10);
    }

    .score-box .k {
        color: #a7f3d0;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.25rem;
    }

    .score-box .v {
        color: #ffffff;
        font-size: 1.2rem;
        font-weight: 850;
    }

    .info-pill {
        display: inline-block;
        padding: 0.38rem 0.65rem;
        border-radius: 999px;
        background: rgba(148, 163, 184, 0.12);
        border: 1px solid rgba(148, 163, 184, 0.14);
        color: #cbd5e1;
        font-size: 0.82rem;
        font-weight: 650;
        margin: 0.15rem 0.22rem 0.15rem 0;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(15, 23, 42, 0.42);
        padding: 0.45rem;
        border-radius: 18px;
        border: 1px solid rgba(148, 163, 184, 0.12);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 14px;
        color: #cbd5e1;
        font-weight: 700;
        padding: 0.65rem 1rem;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(34, 197, 94, 0.18);
        color: #f8fafc;
    }

    div[data-testid="stFileUploader"] {
        border: 1px dashed rgba(74, 222, 128, 0.36);
        border-radius: 22px;
        padding: 0.55rem;
        background: rgba(15, 23, 42, 0.32);
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.50);
        border: 1px solid rgba(148, 163, 184, 0.14);
        padding: 1rem;
        border-radius: 18px;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff;
        font-weight: 850;
    }

    .stAlert {
        border-radius: 18px;
    }

    @media (max-width: 900px) {
        .hero-grid, .score-row {
            grid-template-columns: 1fr;
        }
        .hero {
            padding: 1.45rem;
        }
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Memuat model klasifikasi dengan Keras + OpenVINO...")
def load_ecoscan_model(model_path: Path):
    if not model_path.exists():
        st.error(f"File model tidak ditemukan: {model_path}")
        st.stop()

    try:
        import keras
        return keras.saving.load_model(model_path, compile=False)
    except Exception as exc:
        st.error("Model gagal dimuat.")
        st.write("Kemungkinan penyebab: package Keras/OpenVINO belum terpasang benar, atau file model masih memakai komponen TensorFlow khusus.")
        st.code(str(exc), language="text")
        st.stop()


def preprocess_image(uploaded_image: Image.Image) -> np.ndarray:
    image = ImageOps.exif_transpose(uploaded_image)
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def predict_image(model, image: Image.Image, threshold: float):
    processed = preprocess_image(image)
    raw_output = model.predict(processed, verbose=0)
    raw_score = float(np.asarray(raw_output).ravel()[0])

    if raw_score > threshold:
        return {
            "raw_score": raw_score,
            "label_code": "R",
            "label_name": "Anorganik",
            "english_name": "Inorganic",
            "confidence": raw_score,
            "threshold": threshold,
        }

    return {
        "raw_score": raw_score,
        "label_code": "O",
        "label_name": "Organik",
        "english_name": "Organic",
        "confidence": 1.0 - raw_score,
        "threshold": threshold,
    }


def render_bar(label: str, value: float):
    value = float(min(max(value, 0.0), 1.0))
    st.progress(value, text=f"{label}: {value:.2%}")


def result_card(result):
    card_class = "result-organic" if result["label_code"] == "O" else "result-inorganic"
    icon = "🌿" if result["label_code"] == "O" else "🧴"
    interpretation = (
        "Objek pada gambar lebih dekat dengan karakteristik sampah organik."
        if result["label_code"] == "O"
        else "Objek pada gambar lebih dekat dengan karakteristik sampah anorganik."
    )

    st.markdown(
        f"""
        <div class="{card_class}">
            <div class="result-label">{icon} {result['label_name']}</div>
            <div class="result-sub">Kelas model: {result['label_code']} / {result['english_name']}</div>
            <div class="score-row">
                <div class="score-box">
                    <div class="k">Confidence</div>
                    <div class="v">{result['confidence']:.2%}</div>
                </div>
                <div class="score-box">
                    <div class="k">Skor R / Anorganik</div>
                    <div class="v">{result['raw_score']:.4f}</div>
                </div>
            </div>
            <div style="margin-top: 1rem; color: #e5e7eb; line-height: 1.55;">{interpretation}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown("### ⚙️ Pengaturan Model")
    threshold = st.slider(
        "Threshold kelas Anorganik/R",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.01,
        help="Jika skor model > threshold, gambar diklasifikasikan sebagai Anorganik/R.",
    )

    st.markdown("---")
    st.markdown("### 🧭 Mapping Kelas")
    st.markdown("<span class='info-pill'>O = Organik</span>", unsafe_allow_html=True)
    st.markdown("<span class='info-pill'>R = Anorganik</span>", unsafe_allow_html=True)
    st.caption("Output model berupa satu nilai sigmoid. Nilai mendekati 1 mengarah ke kelas R/Anorganik.")

    st.markdown("---")
    st.markdown("### 🧩 Runtime")
    st.caption("Python 3.14.x")
    st.caption("Keras backend: OpenVINO")

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">♻️ AI Waste Classification System</div>
        <h1>EcoScan<br/>Klasifikasi Sampah</h1>
        <p>
            Aplikasi berbasis Streamlit untuk mengidentifikasi gambar sampah menjadi kategori organik atau anorganik
            menggunakan model transfer learning MobileNetV2.
        </p>
        <div class="hero-grid">
            <div class="mini-stat"><div class="num">224×224</div><div class="label">Ukuran input gambar model</div></div>
            <div class="mini-stat"><div class="num">87,82%</div><div class="label">Akurasi pengujian dari notebook</div></div>
            <div class="mini-stat"><div class="num">97,09%</div><div class="label">AUC pengujian dari notebook</div></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

model = load_ecoscan_model(MODEL_PATH)

tab_prediksi, tab_panduan, tab_evaluasi = st.tabs(["🔍 Prediksi", "📘 Panduan", "📊 Evaluasi Model"])

with tab_prediksi:
    left_col, right_col = st.columns([0.95, 1.05], gap="large")

    with left_col:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Upload Gambar Sampah</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-desc'>Unggah gambar JPG, JPEG, atau PNG. Hasil terbaik diperoleh jika objek sampah tampak jelas dan tidak terlalu banyak objek bercampur.</div>",
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Pilih file gambar",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        image = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang diunggah", use_container_width=True)
        else:
            st.markdown(
                """
                <div class="soft-card">
                    <b>Tips gambar:</b><br/>
                    Ambil gambar dengan cahaya cukup, objek berada di tengah, dan latar belakang tidak terlalu ramai.
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Hasil Analisis</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-desc'>Aplikasi akan menampilkan kelas prediksi, confidence, skor mentah model, dan distribusi probabilitas.</div>",
            unsafe_allow_html=True,
        )

        if image is None:
            st.info("Upload gambar terlebih dahulu untuk menjalankan prediksi.")
        else:
            result = predict_image(model, image, threshold)
            result_card(result)

            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Distribusi Skor</div>", unsafe_allow_html=True)
            render_bar("Anorganik/R", result["raw_score"])
            render_bar("Organik/O", 1.0 - result["raw_score"])

            c1, c2, c3 = st.columns(3)
            c1.metric("Threshold", f"{result['threshold']:.2f}")
            c2.metric("Skor Mentah", f"{result['raw_score']:.4f}")
            c3.metric("Confidence", f"{result['confidence']:.2%}")

        st.markdown("</div>", unsafe_allow_html=True)

with tab_panduan:
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Cara Kerja Aplikasi</div>", unsafe_allow_html=True)
        st.write(
            "Aplikasi memuat model `waste_classifier_final.keras`, lalu gambar yang diunggah dikonversi ke RGB, "
            "diubah ukurannya menjadi 224 × 224 piksel, dan dinormalisasi dengan pembagian 255. Tahapan ini mengikuti preprocessing pada notebook training."
        )
        st.markdown(
            "<span class='info-pill'>RGB</span><span class='info-pill'>Resize 224×224</span><span class='info-pill'>Rescale 1/255</span><span class='info-pill'>Sigmoid</span>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Aturan Interpretasi</div>", unsafe_allow_html=True)
        st.write(
            "Model menghasilkan satu skor sigmoid. Jika skor lebih besar dari threshold, sistem memilih kelas R/Anorganik. "
            "Jika skor lebih kecil atau sama dengan threshold, sistem memilih kelas O/Organik."
        )
        st.code(
            """if skor_model > threshold:
    prediksi = "R / Anorganik"
else:
    prediksi = "O / Organik""",
            language="python",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Batasan Penggunaan</div>", unsafe_allow_html=True)
    st.write(
        "Prediksi dapat kurang stabil jika gambar terlalu gelap, objek sangat kecil, sampah saling bercampur, atau jenis sampah tidak banyak terwakili dalam data latih. "
        "Untuk penggunaan akademik, bagian ini dapat ditulis sebagai batasan implementasi sistem."
    )
    st.markdown("</div>", unsafe_allow_html=True)

with tab_evaluasi:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Ringkasan Evaluasi Model</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-desc'>Angka berikut berasal dari hasil evaluasi model pada notebook.</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test Accuracy", "87,82%")
    c2.metric("Test AUC", "97,09%")
    c3.metric("Test Loss", "0,2909")
    c4.metric("Input Size", "224×224")

    st.markdown("</div>", unsafe_allow_html=True)

    col_cm, col_table = st.columns([0.9, 1.1], gap="large")

    with col_cm:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Confusion Matrix</div>", unsafe_allow_html=True)
        if CONFUSION_PATH.exists():
            st.image(str(CONFUSION_PATH), caption="Confusion Matrix dari notebook", use_container_width=True)
        else:
            st.warning("File confusion_matrix.png tidak ditemukan di folder assets.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_table:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Detail Jumlah Prediksi</div>", unsafe_allow_html=True)
        st.dataframe(
            {
                "Actual": ["O / Organik", "O / Organik", "R / Anorganik", "R / Anorganik"],
                "Predicted": ["O / Organik", "R / Anorganik", "O / Organik", "R / Anorganik"],
                "Jumlah": [1367, 34, 272, 840],
            },
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
