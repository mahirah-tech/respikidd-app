import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import noisereduce as nr
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from datetime import datetime

# Pengaturan Halaman Ramah Anak
st.set_page_config(page_title="RespiKidd - Sahabat Napas Si Kecil", page_icon="🧸", layout="centered")

# --- CUSTOM CSS UNTUK TAMPILAN CERIA & RAMAH ANAK ---
st.markdown("""
<style>
    /* Latar belakang biru langit pastel yang menenangkan */
    .stApp {
        background: linear-gradient(180deg, #E3F4FE 0%, #FFF9E6 100%);
    }
    /* Kartu sambutan maskot */
    .kid-banner {
        background-color: #FFFFFF;
        border: 3px solid #7BD3EA;
        border-radius: 25px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(123, 211, 234, 0.25);
        margin-bottom: 20px;
    }
    .kid-banner h1 {
        color: #FF6B6B;
        font-family: 'Comic Sans MS', 'Trebuchet MS', sans-serif;
        margin-bottom: 5px;
    }
    .kid-banner p {
        color: #4A5568;
        font-size: 16px;
    }
    /* Tombol utama besar, bulat, dan berwarna cerah */
    div.stButton > button:first-child {
        background-color: #FF8E3C;
        color: white;
        border-radius: 25px;
        border: none;
        padding: 12px 24px;
        font-size: 18px;
        font-weight: bold;
        box-shadow: 0 4px 10px rgba(255, 142, 60, 0.4);
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #FF6B6B;
        transform: scale(1.02);
    }
    /* Kotak konten bulat */
    div[data-testid="stExpander"] {
        background-color: #FFFFFF;
        border-radius: 18px;
        border: 2px solid #F6D6D6;
    }
</style>
""", unsafe_allow_html=True)

# Inisialisasi Riwayat di Memori Sesi
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

# --- BANNER MASKOT RAMAH ANAK ---
st.markdown("""
<div class="kid-banner">
    <h1>🧸🩺 Halo, Aku RespiKidd!</h1>
    <p>Yuk, periksa suara batuk si kecil bersama <b>Dokter Beruang</b>! Cepat, seru, dan tanpa rasa sakit ✨</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎤 Cek Suara Si Kecil", "🌟 Buku Bintang (Riwayat)", "🎈 Pojok Belajar Bunda & Ayah"])

# ==========================================
# TAB 1: SKRINING BATUK & GEJALA KLINIS
# ==========================================
with tab1:
    st.markdown("### 🐣 1. Kenalan Dulu Yuk!")
    col1, col2 = st.columns(2)
    with col1:
        nama_anak = st.text_input("Nama Panggilan Si Kecil:", placeholder="Contoh: Adik Bintang 🌟")
    with col2:
        usia_bulan = st.number_input("Usia Si Kecil (Bulan):", min_value=1, max_value=36, value=12)

    gejala = st.multiselect(
        "Apakah si kecil merasakan hal ini juga? (Boleh pilih lebih dari satu):",
        ["🤒 Badan hangat / Demam", " whistles / Bunyi napas 'ngik-ngik' (Mengi)", "💨 Napas terasa cepat", "💧 Banyak dahak / lendir"]
    )
    asap_rokok = st.radio("Apakah ada asap rokok di sekitar rumah?", ["Tidak ada, udara bersih! 🌿", "Ya, ada yang merokok 🚬"], horizontal=True)

    st.markdown("### 🎙️ 2. Rekam Suara Batuk ke Mikrofon Ajaib!")
    st.info("💡 **Tips untuk Ayah/Bunda:** Ajak si kecil bermain sambil mendekatkan HP sekitar 10–15 cm saat anak batuk ya!")
    
    audio_rekam = st.audio_input("Tekan tombol mikrofon merah untuk mulai merekam:")
    audio_upload = st.file_uploader("Atau pilih rekaman suara dari galeri HP (WAV/MP3):", type=["wav", "mp3", "ogg"])

    audio_source = audio_rekam if audio_rekam is not None else audio_upload

    if st.button("✨ Periksa Suara Bersama Dokter Beruang! ✨", use_container_width=True):
        if audio_source is None:
            st.warning("🐻 Wah, suara batuknya belum masuk nih! Yuk rekam atau unggah suaranya dulu ya.")
        else:
            with st.spinner("🐻 Dokter Beruang sedang mendengarkan dan menggambar gelombang suara..."):
                # Proses Audio
                y, sr = librosa.load(audio_source, sr=22050)
                y_clean = nr.reduce_noise(y=y, sr=sr)
                y_norm = librosa.util.normalize(y_clean)

                S = librosa.feature.melspectrogram(y=y_norm, sr=sr, n_mels=128)
                S_dB = librosa.power_to_db(S, ref=np.max)

                # Buat Grafik Mel-Spectrogram Cerah (viridis)
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

                # Animasi Balon Hadiah Keberanian!
                st.balloons()
                st.success(f"🎉 Hebat sekali {nama_anak or 'Si Kecil'}! Ini bintang keberanian untukmu ⭐")

                st.divider()
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
                        "Suara batuk dan gejala si kecil menunjukkan tanda-tanda yang mirip dengan **Bronkitis**. "
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
                    "Nama Si Kecil": nama_anak or "Adik Hebat",
                    "Usia (Bln)": usia_bulan,
                    "Skor Bronkitis": f"{prob_bronkitis*100:.1f}%",
                    "Pesan Dokter": status
                })

# ==========================================
# TAB 2: RIWAYAT PEMERIKSAAN
# ==========================================
with tab2:
    st.subheader("🌟 Buku Bintang Pemeriksaan Si Kecil")
    st.write("Catatan ini bisa Ayah & Bunda tunjukkan langsung kepada dokter saat periksa:")
    if len(st.session_state.riwayat) == 0:
        st.info("Belum ada catatan bintang hari ini. Yuk cek suara si kecil di tab pertama!")
    else:
        st.dataframe(st.session_state.riwayat, use_container_width=True)

# ==========================================
# TAB 3: PUSAT EDUKASI & KALKULATOR TAKIPNEA
# ==========================================
with tab3:
    st.subheader("🎈 Pojok Belajar Bunda & Ayah")
    
    with st.expander("🐻 1. Kenapa Saluran Napas Balita Mudah Teriritasi?", expanded=True):
        st.write(
            "Saluran napas anak di bawah 3 tahun masih mungil dan lembut! Saat terkena virus atau asap rokok, "
            "dinding saluran napas bisa membengkak dan menghasilkan lendir (bronkitis). "
            "Menjaga rumah bebas asap rokok adalah hadiah terbaik untuk paru-paru si kecil 🌿."
        )
        
    with st.expander("🎵 2. Membedakan Bunyi Batuk Biasa vs Bronkitis"):
        st.markdown(
            "- **Batuk Flu Biasa:** Terdengar ringan di tenggorokan, anak tetap ceria bermain.\n"
            "- **Batuk Bronkitis:** Terdengar **lebih dalam, berat, dan basah**, kadang disertai suara siulan halus seperti peluit kecil saat mengembuskan napas (**Mengi / Wheezing**)."
        )
        
    with st.expander("🚑 3. Tanda Si Kecil Harus Segera ke Dokter / IGD"):
        st.markdown(
            "Segera bawa ke fasilitas kesehatan terdekat jika melihat tanda ini:\n"
            "- Kulit di sela tulang rusuk atau bawah leher tampak **tertarik cekung ke dalam** saat menarik napas.\n"
            "- Bibir atau ujung jari tampak **pucat atau kebiruan**.\n"
            "- Si kecil tampak sangat lemas, terus mengantuk, atau menolak menyusu/minum."
        )

    st.divider()
    st.subheader("⏱️ Permainan Hitung Napas 1 Menit (Cek Takipnea)")
    st.write("Hitung berapa kali perut/dada si kecil naik-turun selama 60 detik saat ia sedang tenang:")
    usia_kat = st.selectbox("Pilih Usia Si Kecil:", ["Kurang dari 2 bulan", "2 – 12 bulan", "1 – 3 tahun"])
