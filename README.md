# Klasifikasi Gambar Anjing dan Kucing Menggunakan CNN

Proyek ini mengimplementasikan Convolutional Neural Network (CNN) untuk mengklasifikasikan gambar anjing dan kucing menggunakan TensorFlow dan Keras.

## Deskripsi

Program ini menggunakan arsitektur CNN sederhana untuk membedakan gambar anjing dan kucing. Model dilatih menggunakan dataset gambar yang terdiri dari dua kelas (anjing dan kucing).

### Fitur

- Implementasi CNN dengan TensorFlow/Keras
- Augmentasi data untuk meningkatkan performa model
- Evaluasi model menggunakan dataset testing
- Prediksi gambar individual

## Persyaratan

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Jupyter Notebook

## Struktur Proyek

```
.
├── dataset/
│   ├── training_set/
│   │   ├── dogs/
│   │   └── cats/
│   └── test_set/
│       ├── dogs/
│       └── cats/
├── main.ipynb
└── README.md
```

## Cara Penggunaan

1. Pastikan semua library yang diperlukan sudah terinstall
2. Siapkan dataset dalam struktur folder yang sesuai
3. Buka file `main.ipynb` menggunakan Jupyter Notebook
4. Jalankan setiap sel secara berurutan

## Arsitektur Model

Model CNN yang digunakan terdiri dari:

1. Input Layer (128x128x3)
2. Convolution Layer dengan 32 filter
3. MaxPooling Layer
4. Convolution Layer kedua
5. MaxPooling Layer kedua
6. Flatten Layer
7. Dense Layer dengan 128 unit
8. Output Layer dengan fungsi aktivasi sigmoid

## Dataset

Dataset harus diorganisir dalam struktur berikut:

- training_set/
  - dogs/
  - cats/
- test_set/
  - dogs/
  - cats/

## Preprocessing Data

- Rescaling gambar (1/255)
- Augmentasi data training:
  - Random shearing
  - Random zooming
  - Horizontal flip

## Pelatihan Model

Model dilatih dengan parameter berikut:

- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Metrics: Accuracy
- Epochs: 50
- Batch Size: 32

## Evaluasi

Model menghitung jumlah prediksi untuk masing-masing kelas (anjing dan kucing) dari dataset testing.

## Kontribusi

Silakan berkontribusi untuk meningkatkan proyek ini. Beberapa area yang dapat ditingkatkan:

1. Peningkatan arsitektur model
2. Optimisasi hyperparameter
3. Penambahan fitur visualisasi
4. Peningkatan dokumentasi

## Lisensi

[MIT License](https://opensource.org/licenses/MIT)
# Convolutional-Neural-Network
