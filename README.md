# Vision Transformer vs Swin Transformer on CIFAR-10  
Perbandingan performa dua arsitektur Vision Transformer — **ViT Small Patch16** dan **Swin Tiny Patch4 Window7** — dalam tugas klasifikasi gambar pada dataset CIFAR-10.

Repository ini berisi notebook `.ipynb` yang mengimplementasikan:
- Preprocessing CIFAR-10
- Fine-tuning dua model Transformer
- Training + Validation loop
- Evaluasi (classification report, confusion matrix)
- Pengukuran inference time
- Visualisasi learning curves
- Perbandingan parameter & performa

---

## 🚀 Cara Menjalankan Kode

### 1. Buka Google Colab
Notebook ini dirancang untuk berjalan di Google Colab dengan GPU.
1. Buka file `.ipynb`
2. Pilih **Runtime → Change runtime type**
3. Pada Hardware accelerator → pilih **GPU (T4)**

## 2. Install Dependencies
Notebook menggunakan library berikut:

```python
!pip install timm matplotlib seaborn scikit-learn
```

## 3. Load Dataset CIFAR-10
Dataset otomatis akan di-download oleh PyTorch:
```
from torchvision import datasets, transforms

train_ds = datasets.CIFAR10(
    root="/content", train=True, download=True, transform=tf_train
)

test_ds = datasets.CIFAR10(
    root="/content", train=False, download=True, transform=tf_test
)
```

## 4. Pilih Model yang Akan Dijalankan
Daftar model yang dibandingkan:
```
MODELS = [
    "vit_small_patch16_224",
    "swin_tiny_patch4_window7_224"
]
```

## 5. Set Hyperparameters
```
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 1 (Bebas tergantung keinginan pengguna)
LR = 3e-4
```

## 6. Training Model
```
history = train(model)
```

## 7. Evaluasi Model
```
report, cm = evaluate(model)
```

## 8. Inferensi dan Pengukuran Waktu
Untuk menghitung latency:
```
ms, fps = inference_time(model)
```

## 9. Visualisasi Output
Notebook otomatis menampilkan:
1. Kurva Loss (train vs val)
2. Kurva Accuracy (train vs val)
3. Confusion Matrix
4. Ringkasan hasil dalam DataFrame
