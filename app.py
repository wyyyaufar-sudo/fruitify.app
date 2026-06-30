import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. SETTING HALAMAN UTAMA WEB & SIDEBAR (UPGRADED ✨)
st.set_page_config(page_title="Deteksi Kualitas Buah AI", layout="centered")
st.title("🍎 Fruitify: Deteksi Kesegaran Buah & Sayur")

# Tambahan Sidebar Informasi Project & SDGs untuk Poin Plus Kelompok
st.sidebar.title("🥑 Tentang Aplikasi")
st.sidebar.info(
    """
    **Sistem Deteksi Kualitas Buah**
    Dibuat oleh: 1. Muhammad Aufar Saputro 2.Muhammad Fadhel Fahrezy 3. Sharim Dawnika Sigit 4. Ali Muhammad
    
    Aplikasi berbasis Deep Learning (MobileNetV2) ini dirancang untuk mendeteksi kesegaran buah secara otomatis guna mendukung gerakan **SDG 12 (Responsible Consumption and Production)** dengan menekan angka pembuangan makanan (*food waste*).
    """
)
st.sidebar.title("📊 Ringkasan Model")
st.sidebar.write("• **Akurasi Validasi:** 97.60%")
st.sidebar.write("• **Akurasi Pengujian:** 94.77%")
st.sidebar.write("• **Total Kelas Model:** 18 Kategori")

# 2. LOAD MODEL AI (.h5) YANG SUDAH KAMU DOWNLOAD
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('model_food_waste_v2.h5')

try:
    model = load_my_model()
    st.sidebar.success("🤖 Model AI Berhasil Dimuat!")
except Exception as e:
    st.sidebar.error(f"⚠️ Gagal memuat model. Error: {e}")

# 3. DAFTAR 18 LABEL KELAS (Pastikan urutannya sudah disamakan dengan Colab)
LABELS = [
    'freshapples', 'freshbanana', 'freshbittergroud', 'freshcapsicum', 'freshcucumber', 'freshokra', 'freshoranges', 'freshpotato', 'freshtomato', 'rottenapples', 'rottenbanana', 'rottenbittergroud', 'rottencapsicum', 'rottencucumber', 'rottenokra', 'rottenoranges', 'rottenpotato', 'rottentomato'
]

# 4. FITUR UPLOAD GAMBAR DI WEB
uploaded_file = st.file_uploader("Pilih atau Drop foto buah/sayur di sini...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Menampilkan gambar yang diupload user
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)
    
    with st.spinner("🔄 Sedang menganalisis piksel gambar..."):
        # 5. PREPROCESSING GAMBAR AGAR COCOK DENGAN MOBILENETV2
        img_resized = image.resize((224, 224)) 
        img_array = np.array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        # 6. PROSES PREDIKSI OLEH AI
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0]) # Mengubah output jadi nilai probabilitas
        
        predicted_class_idx = np.argmax(predictions[0])
        predicted_label = LABELS[predicted_class_idx]
        confidence = np.max(score) * 100
        
        # Merapikan teks label agar tidak dempet (misal: 'freshapples' jadi 'Fresh Apples')
        clean_label = predicted_label.replace('fresh', 'Fresh ').replace('rotten', 'Rotten ').title()

    # 7. TAMPILKAN HASILNYA DI LAYAR WEB (UPGRADED INTERAKTIF ✨)
    st.subheader("📊 Hasil Analisis AI:")
    
    # Kondisi Jika Buah Segar vs Busuk
    if "Rotten" in clean_label:
        st.error(f"### Hasil Prediksi: {clean_label} (Kondisi Kurang Baik)")
        st.metric(label="Tingkat Keyakinan AI", value=f"{confidence:.2f}%")
        st.write("💡 **Rekomendasi Tindakan (SDGs):** Segera pisahkan dari buah yang segar agar pembusukan tidak menular. Buah ini sangat cocok dialihkan menjadi pupuk kompos organik atau pakan ternak untuk mengurangi limbah organik.")
    else:
        st.success(f"### Hasil Prediksi: {clean_label} (Kondisi Baik)")
        st.metric(label="Tingkat Keyakinan AI", value=f"{confidence:.2f}%")
        st.write("💡 **Rekomendasi Tindakan (SDGs):** Buah dalam kondisi prima! Siap dikonsumsi langsung, diproses menjadi bahan pangan matang, atau didistribusikan ke pasar sebelum kualitasnya menurun.")

    # FITUR BONUS: Menampilkan Top 3 Probabilitas Teratas agar web terlihat canggih
    st.write("---")
    st.write("📋 **Analisis Detail Probabilitas (Top 3):**")
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    for i in top_3_idx:
        proba_score = tf.nn.softmax(predictions[0])[i] * 100
        label_nama = LABELS[i].replace('fresh', 'Fresh ').replace('rotten', 'Rotten ').title()
        st.write(f"- {label_nama}: **{proba_score:.2f}%**")