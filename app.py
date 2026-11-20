import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import noisereduce as nr
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from datetime import datetime

# Pengaturan Halaman
st.set_page_config(page_title="RespiKidd - Sahabat Napas Si Kecil", page_icon="🧸", layout="centered")

# --- CUSTOM CSS: KONTRAS TINGGI (ANTI-DARK MODE BUG) & SUPER GEMOY ---
st.markdown("""
<style>
    /* Paksa latar belakang pastel cerah & teks gelap agar kontras 100% terbaca */
    .stApp {
        background: linear-gradient(180deg, #DDF2FD 0%, #FEF6E4 100%) !important;
        color: #1E293B !important;
    }
    
    /* Paksa semua label, judul, paragraf, dan teks widget berwarna gelap tajam */
    h1, h2, h3, h4, p, span, label, div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stWidgetLabel"] p, .stRadio label, .stCheckbox label {
        color: #1E293B !important;
    }

    /* Kotak input (Text, Number, Selectbox) berlatar putih bersih dengan teks hitam */
    input, textarea, select, div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 2px solid #9BBEC8 !important;
        border-radius: 15px !important;
    }

    /* Banner Maskot Utama */
    .kid-banner {
        background-color: #FFFFFF;
        border: 4px solid #5AB2FF;
        border-radius: 28px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 10px 20px rgba(90, 178, 255, 0.25);
        margin-bottom: 20px;
    }
    .kid-banner h1 {
        color: #E11D48 !important;
        font-family: 'Trebuchet MS', sans-serif;
        font-size: 32px !important;
        margin-bottom: 8px;
    }
    .kid-banner p {
        color: #334155 !important;
        font-size: 16px !important;
        font-weight: 500;
    }

    /* Kartu Putih Pembungkus Konten agar tidak menyatu dengan background */
    .gemoy-card {
        background-color: #FFFFFF;
        border-radius: 22px;
        padding: 18px 22px;
        border: 3px solid #FFD1DA;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.06);
        margin-bottom: 16px;
    }

    /* Tombol Utama Besar, Bulat, dan Teks Putih Kontras */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #FF6B6B 0%, #FF8E3C 100%) !important;
        color: #FFFFFF !important;
        border-radius: 30px !important;
        border: 3px solid #FFFFFF !important;
        padding: 14px 28px !important;
        font-size: 18px !important;
        font-weight: 800 !important;
        box-shadow: 0 6px 15px rgba(255, 107, 107, 0.45) !important;
    }
    div.stButton > button:first-child p {
        color: #FFFFFF !important;
    }
    div.stButton > button:first-child:hover {
        transform: scale(1.03);
    }

    /* Tab Menu agar jelas terbaca */
    button[data-baseweb="tab"] {
        background-color: #FFFFFF !important;
        border-radius: 15px 15px 0 0 !important;
        margin-right: 4px !important;
        border: 2px solid #CBD5E1 !important;
    }
    button[data-baseweb="tab"] p {
        color: #0F172A !important;
        font-weight: bold !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #5AB2FF !important;
        border-color: #5AB2FF !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #FFFFFF !important;
    }

    /* Expander Edukasi Putih Bersih */
    div[data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border-radius: 18px !important;
        border: 2px solid #93C5FD !important;
    }
    
    /* Metric / Angka Hasil */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 18px;
        border: 2px solid #CBD5E1;
    }
    div[data-testid="stMetricValue"] > div {
        color: #0F172A !important;
    }
</style>
""", unsafe_allow_html=True)

# Inisialisasi Riwayat
if "riwayat" not in st.session_state:
    st.session_state.riwayat = []

# --- MODEL CNN ---
@st.cache_resource
def load_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

cnn_model = load_cnn_model()

# --- BANNER MASKOT ---
st.markdown("""
<div class="kid-banner">
    <h1>🧸🩺 Halo, Aku Dokter Beruang!</h1>
    <p>Selamat datang di <b>RespiKidd</b>! Yuk periksa suara batuk si kecil dengan cara yang seru, lembut, dan tanpa rasa takut 🌈✨</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎤 Cek Suara Batuk", "🌟 Buku Bintang", "🎈 Pojok Belajar"])

# ==========================================
# TAB 1: SKRINING BATUK & INTERAKSI ANAK
# ==========================================
with tab1:
    st.markdown('<div class="gemoy-card"><h3>🐣 Langkah 1: Kenalan & Pilih Hadiah Stiker!</h3><p>Biar si kecil makin semangat, pilih dulu stiker hadiah keberaniannya ya:</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        nama_anak = st.text_input("Nama Panggilan Si Kecil:", placeholder="Contoh: Adik Kimi 🌟")
    with col2:
        usia_bulan = st.number_input("Usia Si Kecil (Bulan):", min_value=1, max_value=36, value=12)

    stiker_pilihan = st.radio(
        "Pilih Stiker Hadiah untuk Si Kecil:",
        ["🦖 Dino Pemberani", "🦄 Kuda Poni Ajaib", "🚀 Roket Luar Angkasa", "🐱 Kucing Lucu"],
        horizontal=True
    )

    gejala = st.multiselect(
        "Apakah si kecil merasakan tanda ini juga? (Boleh pilih lebih dari satu):",
        ["🤒 Badan hangat / Demam", " whistles / Bunyi napas 'ngik-ngik' (Mengi)", "💨 Napas terasa cepat", "💧 Banyak dahak / lendir"]
    )
    asap_rokok = st.radio("Apakah ada asap rokok di sekitar rumah?", ["Tidak ada, udara bersih! 🌿", "Ya, ada yang merokok 🚬"], horizontal=True)

    st.markdown('<div class="gemoy-card"><h3>🎙️ Langkah 2: Tiup & Batuk ke Mikrofon Ajaib!</h3><p>Dekatkan HP sekitar 10–15 cm dari mulut si kecil saat batuk, lalu tekan tombol rekam di bawah ini:</p></div>', unsafe_allow_html=True)
    
    audio_rekam = st.audio_input("Tekan ikon mikrofon merah untuk merekam:")
    audio_upload = st.file_uploader("Atau unggah file rekaman suara (WAV/MP3):", type=["wav", "mp3", "ogg"])

    audio_source = audio_rekam if audio_rekam is not None else audio_upload

    if st.button("✨ Periksa Suara Bersama Dokter Beruang! ✨", use_container_width=True):
        if audio_source is None:
            st.warning("🐻 Wah, suara batuknya belum masuk nih! Yuk rekam atau unggah suaranya dulu ya.")
        else:
            with st.spinner("🐻 Dokter Beruang sedang mendengarkan dan melukis gelombang suara..."):
                # Proses Audio
                y, sr = librosa.load(audio_source, sr=22050)
                y_clean = nr.reduce_noise(y=y, sr=sr)
                y_norm = librosa.util.normalize(y_clean)

                S = librosa.feature.melspectrogram(y=y_norm, sr=sr, n_mels=128)
                S_dB = librosa.power_to_db(S, ref=np.max)

                # Buat Grafik Mel-Spectrogram Cerah
                fig, ax = plt.subplots(figsize=(6, 3))
                img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax, cmap='viridis')
                fig.colorbar(img, ax=ax, format='%+2.0f dB')
                ax.set_title(f'Lukisan Gelombang Suara {nama_anak or "Si Kecil"} 🎨')
                plt.tight_layout()

                # Siapkan Tensor untuk CNN
                lebar_waktu = S_dB.shape[1]
                if lebar_waktu < 128:
                    S_resized = np.pad(S_dB, ((0, 0), (0, 128 - lebar_waktu)), mode='constant')
                else:
                    S_resized = S_dB[:, :128]

                pred = cnn_model.predict(S_resized.reshape(1, 128, 128, 1))
                prob_normal = float(pred[0][0])
                prob_bronkitis = float(pred[0][1])

                # Animasi Perayaan!
                st.balloons()
                st.success(f"🎉 Yey! {nama_anak or 'Si Kecil'} hebat sekali! Ini hadiah **{stiker_pilihan}** untukmu! ⭐")

                st.subheader("🎨 Hasil Lukisan Suara & Analisis AI")
                st.pyplot(fig)

                m1, m2 = st.columns(2)
                m1.metric("Suara Batuk Biasa 🌤️", f"{prob_normal*100:.1f}%")
                m2.metric("Indikasi Bronkitis 🩺", f"{prob_bronkitis*100:.1f}%")

                # Penentuan Triase
                if prob_bronkitis > 0.5 or len(gejala) >= 2 or "💨 Napas terasa cepat" in gejala:
                    status = "PERLU DIPERIKSA DOKTER 🏥"
                    st.error(
                        f"🚨 **PESAN DOKTER BERUANG: {status}**\n\n"
                        "Suara batuk dan gejala si kecil menunjukkan tanda yang mirip dengan **Bronkitis**. "
                        "Jangan panik ya Ayah & Bunda, yuk segera ajak si kecil bertemu dokter atau bidan di Puskesmas terdekat agar cepat sehat kembali!"
                    )
                else:
                    status = "AMAN & PANTAU DI RUMAH 🏡"
                    st.success(
                        f"🌈 **PESAN DOKTER BERUANG: {status}**\n\n"
                        "Hore! Suara batuk si kecil terdengar seperti batuk ringan biasa. "
                        "Terus berikan ASI/air putih hangat, pelukan hangat, dan jauhkan dari asap rokok ya!"
                    )

                # Simpan ke Riwayat
                st.session_state.riwayat.append({
                    "Waktu": datetime.now().strftime("%d-%m-%Y %H:%M"),
