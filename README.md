# PROYEK_UAS_ML_KELOMPOK5
# 🍕🥩 Pizza vs Steak Image Classifier

## 📌 Deskripsi Proyek
Proyek ini merupakan aplikasi berbasis web yang dikembangkan menggunakan **Streamlit** dan **TensorFlow**, yang bertujuan untuk mengklasifikasikan gambar makanan sebagai **Pizza** atau **Steak**. Model CNN (Convolutional Neural Network) digunakan untuk mendeteksi dan memprediksi jenis makanan berdasarkan input gambar dari pengguna.

Aplikasi akan menampilkan hasil klasifikasi beserta tingkat kepercayaan (confidence) dan memberikan peringatan jika model tidak cukup yakin terhadap prediksi yang dihasilkan.

---

## 👥 Tim Kontributor

| Nama Lengkap                 | NIM             |
|-----------------------------|------------------|
| **Nazwa Salsabila**         | 2208107010010    |
| **Berliani Utami**          | 2208107010082    |
| **Raihan Firyal**           | 2208107010084    |
| **Riska Haqika Situmorang** | 2208107010086    |

--


## 🚀 Instruksi Penerapan

### 1. Persiapan Lingkungan
Pastikan Anda telah menginstal Python dan pip, lalu install dependencies berikut:

```bash
pip install streamlit tensorflow opencv-python pillow

Struktur Directory

project/
│
├── food_CNN.h5          # Model hasil training
├── aplikasi.py          # File utama Streamlit
├── README.md            # Dokumentasi proyek
└── pizza_steak(ML)      # File Train Model
