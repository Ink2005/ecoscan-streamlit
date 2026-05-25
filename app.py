import os
from pathlib import Path

# ==========================================================
# EcoScan - Streamlit App untuk Python 3.14.x
# TensorFlow belum tersedia untuk Python 3.14, jadi aplikasi ini
# memakai Keras 3 dengan backend OpenVINO untuk inference.
# Backend harus diset sebelum import keras.
# ==========================================================
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
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        padding: 1.1rem;
        border-radius: 1rem;
        border: 1px solid rgba(49, 51, 63, 0.15);
        background: rgba(250, 250, 250, 0.75);
        min-height: 115px;
    }
    .prediction-box {
        padding: 1.2rem;
        border-radius: 1rem;
        border: 1px solid rgba(49, 51, 63, 0.15);
        background: rgba(250, 250, 250, 0.85);
        margin-top: 1rem;
    }
    .small-note {
        color: #666;
        font-size: 0.92rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Memuat model klasifikasi dengan Keras + OpenVINO...")
def load_ecoscan_model(model_path: Path):
    """Memuat model Keras .keras menggunakan backend OpenVINO untuk Python 3.14."""
    if not model_path.exists():
        st.error(f"File model tidak ditemukan: {model_path}")
        st.stop()

    try:
        import keras
        return keras.saving.load_model(model_path, compile=False)
    except Exception as exc:
        st.error("Model gagal dimuat.")
        st.write("Kemungkinan penyebab: package Keras/OpenVINO belum terpasang benar, atau file model memakai komponen TensorFlow khusus.")
        st.code(str(exc), language="text")
        st.stop()


def preprocess_image(uploaded_image: Image.Image) -> np.ndarray:
    """
    Preprocessing harus sama dengan notebook training:
    - RGB
    - resize 224x224
    - rescale 1/255
    - tambah dimensi batch
    """
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

    # Output sigmoid mendekati 1 berarti R/Inorganic.
    if raw_score > threshold:
        label_code = "R"
        label_name = "Anorganik"
        english_name = "Inorganic"
        confidence = raw_score
    else:
        label_code = "O"
        label_name = "Organik"
        english_name = "Organic"
        confidence = 1.0 - raw_score

    return {
        "raw_score": raw_score,
        "label_code": label_code,
        "label_name": label_name,
        "english_name": english_name,
        "confidence": confidence,
        "threshold": threshold,
    }


def render_probability_bar(prob_inorganic: float):
    prob_inorganic = min(max(float(prob_inorganic), 0.0), 1.0)
    prob_organic = 1.0 - prob_inorganic
    st.progress(prob_inorganic, text=f"Probabilitas Anorganik/R: {prob_inorganic:.2%}")
    st.progress(prob_organic, text=f"Probabilitas Organik/O: {prob_organic:.2%}")


with st.sidebar:
    st.header("Pengaturan")
    threshold = st.slider(
        "Threshold kelas Anorganik/R",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.01,
        help="Jika skor model > threshold, gambar diklasifikasikan sebagai Anorganik/R. Default mengikuti notebook: 0,50.",
    )

    st.divider()
    st.caption("Mapping kelas dari notebook")
    st.write("O = Organik / Organic")
    st.write("R = Anorganik / Inorganic")
    st.write("Output model = sigmoid 1 neuron")
    st.divider()
    st.caption("Runtime")
    st.write("Python 3.14.x")
    st.write("Keras backend: OpenVINO")

st.markdown('<div class="main-title">♻️ EcoScan - Klasifikasi Sampah</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Aplikasi Streamlit untuk mengklasifikasikan gambar sampah menjadi Organik atau Anorganik menggunakan model MobileNetV2.</div>',
    unsafe_allow_html=True,
)

model = load_ecoscan_model(MODEL_PATH)

tab_prediksi, tab_panduan, tab_evaluasi = st.tabs(["Prediksi Gambar", "Panduan Penggunaan", "Evaluasi Model"])

with tab_prediksi:
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("Upload Gambar Sampah")
        uploaded_file = st.file_uploader(
            "Pilih gambar berformat JPG, JPEG, atau PNG",
            type=["jpg", "jpeg", "png"],
        )

        st.markdown(
            "<p class='small-note'>Gunakan gambar yang fokus pada satu objek sampah agar prediksi lebih stabil.</p>",
            unsafe_allow_html=True,
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang diunggah", use_container_width=True)

    with right_col:
        st.subheader("Hasil Prediksi")

        if uploaded_file is None:
            st.info("Upload gambar terlebih dahulu untuk melihat hasil prediksi.")
        else:
            result = predict_image(model, image, threshold)

            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.metric(
                label="Kelas Prediksi",
                value=f"{result['label_name']} ({result['label_code']})",
                delta=f"Confidence {result['confidence']:.2%}",
            )
            st.write(f"Nama kelas Inggris: {result['english_name']}")
            st.write(f"Skor mentah model / probabilitas R: {result['raw_score']:.6f}")
            st.write(f"Threshold yang digunakan: {result['threshold']:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)

            st.write("Distribusi skor:")
            render_probability_bar(result["raw_score"])

            if result["label_code"] == "O":
                st.success("Interpretasi: gambar lebih dekat ke kelas sampah organik.")
            else:
                st.warning("Interpretasi: gambar lebih dekat ke kelas sampah anorganik.")

with tab_panduan:
    st.subheader("Cara Kerja Aplikasi")
    st.write(
        "Aplikasi memuat model `waste_classifier_final.keras` menggunakan Keras 3 dengan backend OpenVINO, "
        "lalu setiap gambar yang diunggah dikonversi ke RGB, diubah ukurannya menjadi 224 × 224 piksel, "
        "dan dinormalisasi dengan pembagian 255. Tahapan ini dibuat sama dengan preprocessing pada notebook training."
    )

    st.subheader("Aturan Interpretasi Output")
    st.write(
        "Model menghasilkan satu nilai sigmoid. Jika nilai tersebut lebih besar dari threshold, gambar diklasifikasikan sebagai R/Anorganik. "
        "Jika nilainya lebih kecil atau sama dengan threshold, gambar diklasifikasikan sebagai O/Organik."
    )

    st.code(
        """if skor_model > 0.5:
    prediksi = "R / Anorganik"
else:
    prediksi = "O / Organik""",
        language="python",
    )

    st.subheader("Catatan Python 3.14")
    st.write(
        "Versi ini tidak memakai TensorFlow. Inference dijalankan lewat Keras 3 + OpenVINO karena TensorFlow belum menyediakan paket resmi untuk Python 3.14. "
        "Jika model gagal dimuat, jalur paling stabil adalah mengubah model ke OpenVINO IR atau memakai Python 3.11/3.12 khusus untuk inference."
    )

    st.subheader("Catatan Penggunaan")
    st.write(
        "Hasil prediksi dapat kurang akurat jika gambar terlalu gelap, objek terlalu kecil, banyak objek bercampur, "
        "atau jenis sampah tidak mirip dengan data latih. Untuk laporan, bagian ini bisa dijadikan batasan implementasi sistem."
    )

with tab_evaluasi:
    st.subheader("Ringkasan Evaluasi dari Notebook")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Test Accuracy", "87,82%")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Test AUC", "97,09%")
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Test Loss", "0,2909")
        st.markdown("</div>", unsafe_allow_html=True)

    st.write(
        "Model sangat baik mengenali kelas organik, tetapi masih memiliki kesalahan pada beberapa sampah anorganik yang diprediksi sebagai organik. "
        "Hal ini terlihat dari recall kelas anorganik yang lebih rendah dibandingkan kelas organik."
    )

    if CONFUSION_PATH.exists():
        st.image(str(CONFUSION_PATH), caption="Confusion Matrix dari notebook", use_container_width=False)

    st.subheader("Confusion Matrix")
    st.dataframe(
        {
            "Actual": ["O / Organik", "O / Organik", "R / Anorganik", "R / Anorganik"],
            "Predicted": ["O / Organik", "R / Anorganik", "O / Organik", "R / Anorganik"],
            "Jumlah": [1367, 34, 272, 840],
        },
        use_container_width=True,
    )
