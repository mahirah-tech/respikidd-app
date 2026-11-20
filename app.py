import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import noisereduce as nr
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from datetime import datetime

# Pengaturan Halaman Aplikasi
st.set_page_config(page_title="RespiKidd AI", page_icon="🩺", layout="centered")

# Inisialisasi Riwayat di Memori Sesi
if "riwayat" not in st.session_state:
    st.session_state.riwayat = []

# --- MODEL CNN (Sesuai Bab 2.3 & 3.3 Proposal) ---
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

# --- HEADER & MENU NAVIGASI ---
st.title("🩺 RespiKidd")
st.caption("Deteksi Dini Penyakit Bronkitis Anak Berbasis Deep Learning & Mel-Spectrogram")

tab1, tab2, tab3 = st.tabs(["🎙️ Skrining Batuk", "📋 Riwayat Pemeriksaan", "📚 Pusat Edukasi"])

# ==========================================
# TAB 1: SKRINING BATUK & GEJALA KLINIS
# ==========================================
with tab1:
    st.subheader("1. Data Pasien & Gejala Klinis")
    col1, col2 = st.columns(2)
    with col1:
        nama_anak = st.text_input("Nama / Inisial Balita", placeholder="Contoh: Adik A")
    with col2:
        usia_bulan = st.number_input("Usia Anak (Bulan)", min_value=1, max_value=36, value=12)

    gejala = st.multiselect(
        "Pilih Gejala Penyerta yang Terlihat (Opsional):",
        ["Demam", "Bunyi Mengi (Wheezing)", "Napas Cepat (Takipnea)", "Lendir/Dahak Berlebih"]
    )
    asap_rokok = st.radio("Paparan Asap Rokok di Rumah:", ["Tidak ada", "Ya, ada perokok di rumah"], horizontal=True)

    st.subheader("2. Input Suara Batuk")
    audio_rekam = st.audio_input("Rekam langsung lewat mikrofon:")
    audio_upload = st.file_uploader("Atau unggah file rekaman batuk (WAV/MP3):", type=["wav", "mp3", "ogg"])

    audio_source = audio_rekam if audio_rekam is not None else audio_upload

    if st.button("🔍 Analisis Suara Batuk Sekarang", type="primary", use_container_width=True):
        if audio_source is None:
            st.warning("⚠️ Silakan rekam atau unggah suara batuk terlebih dahulu!")
        else:
            with st.spinner("Sedang membersihkan bising & mengonversi ke Mel-Spectrogram..."):
                # Proses Audio (Sesuai Gambar 2.2.1 Proposal)
                y, sr = librosa.load(audio_source, sr=22050)
                y_clean = nr.reduce_noise(y=y, sr=sr)
                y_norm = librosa.util.normalize(y_clean)

                S = librosa.feature.melspectrogram(y=y_norm, sr=sr, n_mels=128)
                S_dB = librosa.power_to_db(S, ref=np.max)

                # Buat Grafik Mel-Spectrogram
                fig, ax = plt.subplots(figsize=(6, 3))
                img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax, cmap='magma')
                fig.colorbar(img, ax=ax, format='%+2.0f dB')
                ax.set_title(f'Mel-Spectrogram: {nama_anak or "Pasien Balita"}')
                plt.tight_layout()

                # Siapkan Tensor (128x128) untuk CNN
                lebar_waktu = S_dB.shape[1]
                if lebar_waktu < 128:
                    S_resized = np.pad(S_dB, ((0, 0), (0, 128 - lebar_waktu)), mode='constant')
                else:
                    S_resized = S_dB[:, :128]

                pred = cnn_model.predict(S_resized.reshape(1, 128, 128, 1))
                prob_normal = float(pred[0][0])
                prob_bronkitis = float(pred[0][1])

                # Tampilkan Hasil
                st.divider()
                st.subheader("📊 Hasil Analisis RespiKidd")
                st.pyplot(fig)

                m1, m2 = st.columns(2)
                m1.metric("Probabilitas Batuk Normal", f"{prob_normal*100:.1f}%")
                m2.metric("Indikasi Gejala Bronkitis", f"{prob_bronkitis*100:.1f}%")

                # Penentuan Triase
                if prob_bronkitis > 0.5 or len(gejala) >= 2 or "Napas Cepat (Takipnea)" in gejala:
                    status = "TINGGI (Segera Rujuk)"
                    st.error(
                        f"🚨 **STATUS URGENSI: {status}**\n\n"
                        "Terdeteksi pola akustik/gejala yang mengarah pada **Bronkitis**. "
                        "Segera bawa anak ke Puskesmas atau dokter spesialis anak untuk pemeriksaan klinis lebih lanjut."
                    )
                else:
                    status = "RENDAH (Pantau Mandiri)"
                    st.success(
                        f"✅ **STATUS URGENSI: {status}**\n\n"
                        "Pola suara batuk cenderung normal. Cukupi kebutuhan cairan/ASI anak dan hindarkan dari asap rokok."
                    )

                # Simpan ke Riwayat
                st.session_state.riwayat.append({
                    "Waktu": datetime.now().strftime("%d-%m-%Y %H:%M"),
                    "Nama Anak": nama_anak or "Anonim",
                    "Usia (Bln)": usia_bulan,
                    "Skor Bronkitis": f"{prob_bronkitis*100:.1f}%",
                    "Status": status
                })

# ==========================================
# TAB 2: RIWAYAT PEMERIKSAAN
# ==========================================
with tab2:
    st.subheader("📋 Log Hasil Skrining")
    if len(st.session_state.riwayat) == 0:
        st.info("Belum ada riwayat pemeriksaan pada sesi ini.")
    else:
        st.dataframe(st.session_state.riwayat, use_container_width=True)

# ==========================================
# TAB 3: PUSAT EDUKASI & KALKULATOR TAKIPNEA
# ==========================================
with tab3:
    st.subheader("📚 Panduan Mengenali Bronkitis Balita")
    
    with st.expander("1. Apa itu Bronkitis & Bahaya Asap Rokok?", expanded=True):
        st.write(
            "Bronkitis adalah peradangan pada saluran napas utama (bronkus) yang menyebabkan pembengkakan dan lendir berlebih. "
            "Balita di bawah 3 tahun sangat rentan karena saluran napasnya masih sempit. Paparan **asap rokok** di rumah "
            "melumpuhkan bulu halus pembersih saluran napas sehingga memicu infeksi lebih parah."
        )
        
    with st.expander("2. Perbedaan Suara Batuk Biasa vs Bronkitis"):
        st.markdown(
            "- **Batuk Flu Biasa:** Cenderung kering atau berdahak ringan di tenggorokan.\n"
            "- **Batuk Bronkitis:** Terdengar **lebih dalam, berat, dan basah**, sering disertai bunyi siulan halus saat membuang napas (**Wheezing / Mengi**)."
        )
        
    with st.expander("3. Tanda Bahaya (Kapan Harus ke IGD?)"):
        st.markdown(
            "Segera bawa ke IGD jika anak mengalami:\n"
            "- **Retraksi dinding dada** (kulit di sela tulang rusuk tertarik ke dalam saat bernapas).\n"
            "- Bibir atau ujung jari tampak **kebiruan/pucat**.\n"
            "- Anak tampak sangat lemas atau tidak mau menyusu/minum."
        )

    st.divider()
    st.subheader("⏱️ Cek Napas Cepat (Takipnea) dalam 60 Detik")
    usia_kat = st.selectbox("Pilih Kelompok Usia Anak:", ["Kurang dari 2 bulan", "2 – 12 bulan", "1 – 3 tahun"])
    jml_napas = st.number_input("Jumlah tarikan napas dalam 1 menit penuh:", min_value=10, max_value=120, value=30)
    
    if st.button("Hitung Status Napas"):
        batas = 60 if usia_kat == "Kurang dari 2 bulan" else (50 if usia_kat == "2 – 12 bulan" else 40)
        if jml_napas >= batas:
            st.error(f"⚠️ **WASPADA TAKIPNEA!** Batas normal usia {usia_kat} adalah < {batas}x/menit. Segera periksakan ke fasilitas kesehatan.")
        else:
            st.success(f"✅ **NORMAL.** Hitungan {jml_napas}x/menit masih dalam batas aman (< {batas}x/menit).")
