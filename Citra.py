import matplotlib.pyplot as plt
import numpy as np
from skimage import io, img_as_float, img_as_ubyte, data
from skimage.filters import rank
from skimage.morphology import disk
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# ===============================================================
# 1️⃣ LOAD CITRA (PASTIKAN PATH BENAR)
# ===============================================================
try:
    image_path = r"coksu.jpg"
    image = img_as_float(io.imread(image_path, as_gray=True))  # grayscale agar analisis lebih akurat
    print("✅ Foto berhasil dimuat.")
except Exception as e:
    print(f"⚠️ Gagal memuat foto ({e}), gunakan citra default.")
    image = img_as_float(data.camera())

# ===============================================================
# 2️⃣ TAMBAHKAN NOISE MANUAL (STABIL)
# ===============================================================
def add_salt_pepper_noise(image, amount=0.1, salt_vs_pepper=0.5, seed=42):
    np.random.seed(seed)
    noisy = np.copy(image)
    num_total = image.size
    num_salt = int(num_total * amount * salt_vs_pepper)
    num_pepper = int(num_total * amount * (1 - salt_vs_pepper))

    # Koordinat salt
    coords_salt = (
        np.random.randint(0, image.shape[0], num_salt),
        np.random.randint(0, image.shape[1], num_salt)
    )
    # Koordinat pepper
    coords_pepper = (
        np.random.randint(0, image.shape[0], num_pepper),
        np.random.randint(0, image.shape[1], num_pepper)
    )

    noisy[coords_salt] = 1
    noisy[coords_pepper] = 0
    return noisy

noisy = add_salt_pepper_noise(image, amount=0.1)

# ===============================================================
# 3️⃣ FILTERING
# ===============================================================
noisy_ubyte = img_as_ubyte(noisy)

mean_filtered = rank.mean(noisy_ubyte, footprint=disk(3))
min_filtered = rank.minimum(noisy_ubyte, footprint=disk(3))
median_filtered = rank.median(noisy_ubyte, footprint=disk(3))
max_filtered = rank.maximum(noisy_ubyte, footprint=disk(3))

mean_f = img_as_float(mean_filtered)
min_f = img_as_float(min_filtered)
median_f = img_as_float(median_filtered)
max_f = img_as_float(max_filtered)

# ===============================================================
# 4️⃣ HITUNG METRIK
# ===============================================================
metrics = {
    "Noisy": (psnr(image, noisy), ssim(image, noisy, data_range=1.0)),
    "Mean": (psnr(image, mean_f), ssim(image, mean_f, data_range=1.0)),
    "Min": (psnr(image, min_f), ssim(image, min_f, data_range=1.0)),
    "Median": (psnr(image, median_f), ssim(image, median_f, data_range=1.0)),
    "Max": (psnr(image, max_f), ssim(image, max_f, data_range=1.0)),
}

# ===============================================================
# 5️⃣ VISUALISASI
# ===============================================================
fig, axes = plt.subplots(1, 6, figsize=(16, 5))
titles = ["Original", "Noisy", "Mean", "Min", "Median", "Max"]
images = [image, noisy, mean_f, min_f, median_f, max_f]

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()

# ===============================================================
# 6️⃣ CETAK HASIL PSNR DAN SSIM
# ===============================================================
print("\n=== Evaluasi PSNR & SSIM ===")
for name, (p, s) in metrics.items():
    print(f"{name:<10} PSNR: {p:5.2f}, SSIM: {s:.4f}")

