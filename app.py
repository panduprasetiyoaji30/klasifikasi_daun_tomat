import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Load Model
model = tf.keras.models.load_model("cnn_tomato_model.keras")

# Class Names
class_names = [
    "Tomato__early_blight",
    "Tomato__healthy",
    "Tomato__late_blight",
    "Tomato__leaf_mold",
    "Tomato__septoria_leaf_spot"
]

# Dictionary Nama Penyakit
label_mapping = {
    "Tomato__healthy": "Daun Sehat",
    "Tomato__early_blight": "Early Blight",
    "Tomato__late_blight": "Late Blight",
    "Tomato__leaf_mold": "Leaf Mold",
    "Tomato__septoria_leaf_spot": "Septoria Leaf Spot"
}

# Dictionary Deskripsi
description = {
    "Daun Sehat":
        "Daun tomat berada dalam kondisi sehat dan tidak menunjukkan gejala penyakit.",

    "Early Blight":
        "Penyakit yang disebabkan oleh jamur Alternaria solani. Gejala berupa bercak cokelat dengan pola melingkar pada daun.",

    "Late Blight":
        "Penyakit yang disebabkan oleh Phytophthora infestans. Gejala berupa bercak kehitaman yang cepat menyebar pada daun.",

    "Leaf Mold":
        "Penyakit akibat jamur Passalora fulva yang ditandai dengan bercak kekuningan pada permukaan daun dan lapisan jamur di bagian bawah daun.",

    "Septoria Leaf Spot":
        "Penyakit akibat jamur Septoria lycopersici yang ditandai dengan bercak kecil berwarna cokelat keabu-abuan pada daun."
}

# Dictionary Penanganan
solution = {
    "Daun Sehat":
        "Pertahankan penyiraman, pemupukan, dan kebersihan lahan agar tanaman tetap sehat.",

    "Early Blight":
        "Buang daun yang terinfeksi, lakukan rotasi tanaman, dan gunakan fungisida sesuai anjuran.",

    "Late Blight":
        "Segera buang bagian tanaman yang terinfeksi dan lakukan penyemprotan fungisida untuk mencegah penyebaran.",

    "Leaf Mold":
        "Kurangi kelembapan di sekitar tanaman, tingkatkan sirkulasi udara, dan gunakan fungisida bila diperlukan.",

    "Septoria Leaf Spot":
        "Buang daun yang terserang, hindari penyiraman mengenai daun, dan gunakan fungisida sesuai rekomendasi."
}


st.set_page_config(
    page_title="Klasifikasi Penyakit Daun Tomat",
    page_icon="🍅",
    layout="centered"
)

st.title("🍅 Klasifikasi Penyakit Daun Tomat")
st.write(
    "Aplikasi ini menggunakan model Convolutional Neural Network (CNN) "
    "untuk mengidentifikasi penyakit pada daun tomat berdasarkan citra yang diunggah."
)

uploaded_file = st.file_uploader(
    "Unggah gambar daun tomat",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Gambar yang Diunggah",
        use_container_width=True
    )

    img = image.resize((224, 224))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)


    prediction = model.predict(img, verbose=0)[0]

    for cls, prob in zip(class_names, prediction):
        print(cls, prob)

    index = np.argmax(prediction)
    confidence = prediction[index]

    class_name = class_names[index]
    disease = label_mapping[class_name]

    st.success(f"Hasil Prediksi : **{disease}**")

    st.subheader("Deskripsi")
    st.write(description[disease])

    st.subheader("Saran Penanganan")
    st.write(solution[disease])