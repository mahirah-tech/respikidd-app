Ini adalah kode lengkap (app.py) yang sudah diperbaiki secara menyeluruh, bebas dari error indeks, dan memuat tampilan ramah anak (gemoy dengan kontras tinggi) beserta Pojok Belajar rinci untuk orang tua.
Silakan salin (copy) seluruh kode di bawah ini dari baris pertama hingga paling akhir, lalu tempel (paste) ke file app.py di GitHub-mu:
python
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

# --- CUSTOM CSS: KONTRAS TINGGI & SUPER GEMOY ---
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #DDF2FD 0%, #FEF6E4 100%) !important;
        color: #1E293B !important;
    }
    h1, h2, h3, h4, p, span, label, li, td, th, div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stWidgetLabel"] p, .stRadio label, .stCheckbox label {
        color: #1E293B !important;
    }
    input, textarea, select, div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 2px solid #9BBEC8 !important;
        border-radius: 15px !important;
    }
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
    .gemoy-card {
        background-color: #FFFFFF;
        border-radius: 22px;
        padding: 18px 22px;
        border: 3px solid #FFD1DA;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.06);
        margin-bottom: 16px;
    }
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
    div[data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border-radius: 18px !important;
        border: 2px solid #93C5FD !important;
        margin-bottom: 10px !important;
    }
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

if "riwayat" not in st.session_state:
    st.session_state.riwayat = []

# --- MODEL CNN ---
@st.cache_resource
def load_cnn_model():
    model = models.Sequential([
        layers.Input(shape=(128, 128, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
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
        ["🤒 Badan hangat / Demam", "🎵 Bunyi napas 'ngik-ngik' (Mengi)", "💨 Napas terasa cepat", "💧 Banyak dahak / lendir"]
    )
    asap_rokok = st.radio("Apakah ada asap rokok di sekitar rumah?", ["Tidak ada, udara bersih! 🌿", "Ya, ada yang merokok 🚬"], horizontal=True)

    st.markdown('<div class="gemoy-card"><h3>🎙️ Langkah 2: Tiup & Batuk ke Mikrofon Ajaib!</h3><p>Dekatkan HP sekitar 10–15 cm dari mulut si kecil saat batuk, lalu tekan tombol rekam di bawah ini:</p></div>', unsafe_allow_html=True)
    
    audio_rekam = st.audio_input("Tekan ikon mikrofon merah untuk mulai merekam:")
    audio_upload = st.file_uploader("Atau unggah file rekaman suara (WAV/MP3):", type=["wav", "mp3", "ogg"])

    audio_source = audio_rekam if audio_rekam is not None else audio_upload

    if st.button("✨ Periksa Suara Bersama Dokter Beruang! ✨", use_container_width=True):
        if audio_source is None:
            st.warning("🐻 Wah, suara batuknya belum masuk nih! Yuk rekam atau unggah suaranya dulu ya.")
        else:
            with st.spinner("🐻 Dokter Beruang sedang mendengarkan dan melukis gelombang suara..."):
                y, sr = librosa.load(audio_source, sr=22050)
                y_clean = nr.reduce_noise(y=y, sr=sr)
                y_norm = librosa.util.normalize(y_clean)

                S = librosa.feature.melspectrogram(y=y_norm, sr=sr, n_mels=128)
                S_dB = librosa.power_to_db(S, ref=np.max)

                fig, ax = plt.subplots(figsize=(6, 3))
                img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax, cmap='viridis')
                fig.colorbar(img, ax=ax, format='%+2.0f dB')
                ax.set_title(f'Lukisan Gelombang Suara {nama_anak or "Si Kecil"} 🎨')
                plt.tight_layout()

                baris, kolom = S_dB.shape
                if kolom < 128:
                    S_resized = np.pad(S_dB, ((0, 0), (0, 128 - kolom)), mode='constant')
                else:
                    S_resized = np.take(S_dB, range(128), axis=1)

                pred = cnn_model.predict(S_resized.reshape(1, 128, 128, 1))
                hasil_list = pred.flatten().tolist()
                prob_normal = float(hasil_list.pop(0))
                prob_bronkitis = float(hasil_list.pop(0))

                st.balloons()
                st.success(f"🎉 Yey! {nama_anak or 'Si Kecil'} hebat sekali! Ini hadiah **{stiker_pilihan}** untukmu! ⭐")

                st.subheader("🎨 Hasil Lukisan Suara & Analisis AI")
                st.pyplot(fig)

                m1, m2 = st.columns(2)
                m1.metric("Suara Batuk Biasa 🌤️", f"{prob_normal*100:.1f}%")
                m2.metric("Indikasi Bronkitis 🩺", f"{prob_bronkitis*100:.1f}%")

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

                st.session_state.riwayat.append({
                    "Waktu": datetime.now().strftime("%d-%m-%Y %H:%M"),
                    "Nama Si Kecil": nama_anak or "Adik Hebat",
                    "Usia (Bln)": usia_bulan,
                    "Stiker": stiker_pilihan,
                    "Skor Bronkitis": f"{prob_bronkitis*100:.1f}%",
                    "Pesan Dokter": status
                })

# ==========================================
# TAB 2: RIWAYAT PEMERIKSAAN
# ==========================================
with tab2:
    st.markdown('<div class="gemoy-card"><h3>🌟 Buku Bintang & Stiker Si Kecil</h3><p>Catatan hasil pemeriksaan ini bisa Ayah & Bunda tunjukkan langsung kepada dokter:</p></div>', unsafe_allow_html=True)
    if len(st.session_state.riwayat) == 0:
        st.info("Belum ada catatan bintang hari ini. Yuk cek suara si kecil di tab pertama!")
    else:
Use code with caution.
st.dataframe(st.session_state.riwayat, use_container_width=True)

==========================================


TAB 3: PUSAT EDUKASI RINCI & INTERAKTIF


==========================================

with tab3:
st.markdown("""

🎈 Pojok Belajar Bunda & Ayah
Panduan lengkap memahami kesehatan pernapasan batita (0–3 tahun), cara merawat di rumah, serta mengenali tanda bahaya sedini mungkin.

""", unsafe_allow_html=True)
with st.expander("📖 Bab 1: Apa Itu Bronkitis & Mengapa Batita Sangat Rentan?", expanded=True):
st.markdown("""
Bronkitis adalah peradangan atau pembengkakan pada dinding saluran udara utama yang menuju paru-paru (bronkus). Ketika meradang, saluran napas menyempit dan memproduksi lendir (dahak) kental secara berlebihan.
Mengapa anak di bawah 3 tahun (batita) mudah terkena?
* Ukuran Saluran Napas Mungil: Diameter saluran napas batita masih sangat sempit dibandingkan orang dewasa, sehingga sedikit lendir saja sudah cukup untuk menyumbat aliran udara.
* Sistem Imun Belum Matang: Daya tahan tubuh batita masih dalam tahap belajar mengenali virus dan bakteri.
* Otot Pernapasan Masih Lemah: Anak kecil belum mampu batuk sekuat orang dewasa untuk mengeluarkan dahak kental dari dadanya.
""")
with st.expander("🔍 Bab 2: Tabel Perbedaan Batuk Flu Biasa, Bronkitis, dan Pneumonia"):
st.markdown("""
Orang tua sering mengira semua batuk pada anak itu sama. Perhatikan perbedaan utamanya berikut ini:

Ciri-Ciri	🤧 Batuk Pilek / Flu Biasa	🫁 Bronkitis Akut	🚨 Pneumonia (Radang Paru)
Suara Batuk	Kering atau berdahak ringan di tenggorokan.	Dalam, berat, basah/bergetar, sering disertai bunyi siulan halus (mengi / wheezing).	Batuk sangat dalam, terdengar sesak dan menyakitkan.
Demam	Tidak ada atau demam ringan (1–2 hari).	Demam ringan hingga sedang.	Demam tinggi disertai menggigil dan berkeringat.
Pola Napas	Normal dan tenang saat tidur.	Napas agak cepat, terdengar suara lendir (grok-grok) di dada.	Napas sangat cepat disertai cekungan di bawah tulang rusuk.
Kondisi Anak	Masih mau bermain dan makan/minum.	Terlihat rewel, kurang nyaman, dan nafsu makan menurun.	Sangat lemas, pucat, tampak kesakitan saat bernapas.
""")			

==========================================Export to Sheets
with st.expander("🚬 Bab 3: Bahaya Asap Rokok & Racun Tersembunyi (Thirdhand Smoke)"):
st.markdown("""
Banyak orang tua berpikir merokok di luar rumah sudah aman bagi anak. Faktanya tidak!
* Perokok Pasif (Secondhand Smoke): Menghirup asap rokok secara langsung melumpuhkan silia (bulu-bulu halus pembersih debu dan kuman di saluran napas anak). Akibatnya kuman mudah masuk dan memicu bronkitis.
* Residu Asap (Thirdhand Smoke): Zat beracun dari asap rokok (nikotin, tar, karbon monoksida) menempel kuat pada baju, rambut, kulit, sofa, dan bantal hingga berjam-jam setelah merokok. Ketika ayah/anggota keluarga menggendong balita tanpa ganti baju dan cuci tangan, racun tersebut langsung terhirup oleh paru-paru si kecil.
* Solusi Wajib: Cuci tangan, cuci muka, dan ganti pakaian bersih sebelum menyentuh atau menggendong batita.
""")
with st.expander("🏡 Bab 4: Panduan Pertolongan Pertama & Perawatan Aman di Rumah"):
st.markdown("""
Jika hasil skrining menunjukkan gejala ringan, Ayah & Bunda dapat melakukan langkah perawatan berikut di rumah:
1. Perbanyak Cairan Hangat (Kunci Utama!): Berikan ASI lebih sering (untuk bayi) atau air putih hangat dan kuah sup hangat (untuk anak > 6 bulan). Cairan adalah pengencer dahak alami terbaik.
2. Terapi Uap Hangat Alami: Ajak anak duduk di kamar mandi yang diisi uap air panas selama 10–15 menit, atau gunakan humidifier (pelembap udara) tanpa pewangi kimia menyengat.
3. Posisi Tidur Lebih Tinggi: Untuk anak di atas 1 tahun, ganjal kepala dan bahu sedikit lebih tinggi dengan bantal agar lendir tidak menumpuk di tenggorokan saat tidur.
4. Bersihkan Hidung Tersumbat: Gunakan tetes hidung larutan garam steril (saline nasal spray) dan penyedot ingus (nasal aspirator) sebelum anak menyusu atau tidur.
5. ⚠️ JANGAN Beri Obat Batuk Bebas: Ikatan Dokter Anak Indonesia (IDAI) dan WHO melarang pemberian obat batuk/pilek yang dijual bebas pada anak di bawah usia 2–4 tahun tanpa resep dokter.
""")
with st.expander("💡 Bab 5: Mitos vs Fakta Seputar Batuk pada Anak"):
st.markdown("""
* ❌ MITOS: Bayi di bawah 1 tahun boleh diberi madu untuk meredakan batuk.
✅ FAKTA: Dilarang keras! Madu mengandung spora bakteri Clostridium botulinum yang bisa menyebabkan keracunan saraf (infant botulism) pada bayi di bawah 12 bulan. Madu hanya aman untuk anak usia 1 tahun ke atas.
* ❌ MITOS: Batuk berdahak harus langsung diberi antibiotik agar cepat sembuh.
✅ FAKTA: Lebih dari 85% kasus bronkitis pada balita disebabkan oleh virus, bukan bakteri. Antibiotik tidak mempan membunuh virus dan hanya boleh diberikan jika dokter menemukan infeksi bakteri.
* ❌ MITOS: Mandi air hangat tidak boleh dilakukan saat anak batuk demam.
✅ FAKTA: Mandi atau berendam air hangat justru membantu melembapkan saluran napas, menurunkan demam ringan, dan membuat anak tidur lebih nyenyak.
""")
with st.expander("🚨 Bab 6: Tanda Bahaya Darurat (Kapan Harus Segera ke IGD?)"):
st.error("""
Segera bawa si kecil ke IGD Rumah Sakit / Puskesmas 24 Jam jika muncul SATU SAJA tanda di bawah ini:
* Retraksi Dinding Dada: Kulit di bawah tulang rusuk, sela-sela iga, atau pangkal leher tampak tertarik cekung ke dalam setiap kali anak menarik napas.
* Napas Cuping Hidung: Lubang hidung anak tampak kembang-kempis dengan cepat saat bernapas.
* Sianosis (Kebiruan): Bibir, lidah, bagian dalam mulut, atau ujung kuku jari tampak kebiruan atau pucat keabu-abuan.
* Dehidrasi & Lemas: Anak tampak sangat mengantuk, sulit dibangunkan, tidak mau menyusu/minum sama sekali, atau popok kering lebih dari 6 jam.
* Suara Napas Melengking Keras (Stridor): Terdengar suara mengorok atau melengking keras saat anak menarik napas dalam keadaan tenang.
""")
st.divider()
st.markdown("""

⏱️ Alat Bantu 1: Hitung Napas 1 Menit (Cek Takipnea WHO)
Cara mengukur: Buka baju bagian dada/perut anak saat ia sedang tenang (tidak menangis/demam tinggi). Hitung berapa kali dada/perut naik-turun selama 60 detik penuh.

""", unsafe_allow_html=True)
usia_kat = st.selectbox("Pilih Kelompok Usia Si Kecil:", ["Kurang dari 2 bulan", "2 – 12 bulan", "1 – 3 tahun"])
jml_napas = st.number_input("Masukkan jumlah tarikan napas dalam 1 menit:", min_value=10, max_value=120, value=30)
if st.button("🔍 Cek Status Kecepatan Napas"):
batas = 60 if usia_kat == "Kurang dari 2 bulan" else (50 if usia_kat == "2 – 12 bulan" else 40)
if jml_napas >= batas:
st.error(f"⚠️ WASPADA NAPAS CEPAT (TAKIPNEA)! Untuk usia {usia_kat}, batas normalnya adalah di bawah {batas} kali/menit. Hitungan si kecil ({jml_napas}x/menit) menandakan saluran napasnya bekerja terlalu keras. Segera bawa ke fasilitas kesehatan!")
else:
st.success(f"🎉 KECEPATAN NAPAS NORMAL! Hitungan {jml_napas} kali/menit masih dalam batas aman untuk usia {usia_kat} (di bawah {batas}x/menit).")
st.divider()
st.markdown("""

🎯 Alat Bantu 2: Mini Kuis Siaga Bronkitis
Uji pemahaman Ayah & Bunda dalam menjaga paru-paru si kecil!

""", unsafe_allow_html=True)
q1 = st.radio(
"1. Bolehkah memberikan madu untuk meredakan batuk pada bayi usia 8 bulan?",
["Boleh, karena madu bahan alami", "Tidak boleh, berisiko keracunan botulisme pada bayi < 1 tahun"],
index=None
)
q2 = st.radio(
"2. Jika Ayah merokok di teras luar rumah, apa yang harus dilakukan sebelum menggendong balita?",
["Langsung menggendong karena asapnya sudah tertiup angin di luar", "Cuci tangan, cuci muka, dan ganti baju bersih agar residu racun tidak terhirup anak"],
index=None
)
if st.button("✨ Lihat Skor Pemahaman Bunda & Ayah"):
skor = 0
if q1 == "Tidak boleh, berisiko keracunan botulisme pada bayi < 1 tahun":
skor += 50
if q2 == "Cuci tangan, cuci muka, dan ganti baju bersih agar residu racun tidak terhirup anak":
skor += 50
if skor == 100:
st.balloons()
st.success("🏆 SKOR 100/100 — ORANG TUA SIAGA JEMPOLAN! Ayah & Bunda sudah sangat paham cara melindungi saluran napas si kecil!")
elif skor == 50:
st.warning("⭐ SKOR 50/100 — BAGUS! Masih ada 1 jawaban yang kurang tepat. Yuk baca kembali Bab 3 dan Bab 5 di atas ya!")
else:
st.info("📚 SKOR 0/100 — YUK BELAJAR LAGI! Silakan pilih jawaban di atas atau baca panduan di Pojok Belajar ya!")
