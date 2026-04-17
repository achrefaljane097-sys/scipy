import cv2
import numpy as np
from scipy.fft import dctn, idctn
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

# 📌 1. Charger l'image
image_originale = cv2.imread("moun.jpg", cv2.IMREAD_GRAYSCALE)
image_originale = image_originale.astype(np.float64)

# 📌 2. DCT
def apply_dct_blocks(image):
    h, w = image.shape
    dct_image = np.zeros_like(image)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            bloc = image[i:i+8, j:j+8]
            dct_image[i:i+8, j:j+8] = dctn(bloc, norm='ortho')
    return dct_image

# 📌 3. Watermark
def generer_watermark(taille, cle=42):
    #ajouter seed value
    np.random.seed(cle)
    return np.random.randint(0, 2, taille)

# 📌 4. Insertion QIM
def inserer_qim(dct_image, watermark, delta=30):
    h, w = dct_image.shape
    dct_modif = dct_image.copy()
    idx = 0
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            if idx >= len(watermark):
                break
            coeff = dct_modif[i+3, j+4]
            bit = watermark[idx]

            q = np.round(coeff / delta)
            if bit == 0:
                q = 2 * np.round(q / 2)
            else:
                q = 2 * np.round((q - 1) / 2) + 1

            dct_modif[i+3, j+4] = q * delta
            idx += 1
    return dct_modif

# 📌 5. IDCT
def appliquer_idct_blocks(dct_image):
    h, w = dct_image.shape
    image_rec = np.zeros_like(dct_image)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            bloc = dct_image[i:i+8, j:j+8]
            image_rec[i:i+8, j:j+8] = idctn(bloc, norm='ortho')
    return np.clip(image_rec, 0, 255)

# 📌 6. Attaques
def attaque_bruit(image, sigma=10):
    bruit = np.random.normal(0, sigma, image.shape)
    return np.clip(image + bruit, 0, 255)

def attaque_jpeg(image, qualite=50):
    cv2.imwrite("temp.jpg", image.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, qualite])
    return cv2.imread("temp.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float64)

# 📌 7. Extraction
def extraire_qim(dct_image, taille_wm, delta=30):
    h, w = dct_image.shape
    watermark_extrait = []
    idx = 0
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            if idx >= taille_wm:
                break
            coeff = dct_image[i+3, j+4]
            q = np.round(coeff / delta)
            bit = int(q % 2)
            watermark_extrait.append(abs(bit))
            idx += 1
    return np.array(watermark_extrait)

# =============================
# 🚀 PROGRAMME PRINCIPAL
# =============================

# DCT
dct_image = apply_dct_blocks(image_originale)

# Watermark
taille = (image_originale.shape[0] // 8) * (image_originale.shape[1] // 8)
watermark = generer_watermark(taille)

# Insertion
dct_modif = inserer_qim(dct_image, watermark)

# Image tatouée
image_tatouee = appliquer_idct_blocks(dct_modif)

# (optionnel) attaque
image_attaquee = attaque_bruit(image_tatouee)

# DCT après attaque
dct_attaquee = apply_dct_blocks(image_attaquee)

# Extraction
watermark_extrait = extraire_qim(dct_attaquee, len(watermark))

# 📊 PSNR
psnr_val = psnr(image_originale, image_tatouee, data_range=255)
print(f"PSNR : {psnr_val:.2f} dB")

# 📊 BER
def calculer_ber(wm_original, wm_extrait):
    erreurs = np.sum(wm_original != wm_extrait)
    return erreurs / len(wm_original)

ber_val = calculer_ber(watermark, watermark_extrait)
print(f"BER : {ber_val:.4f}")







# Affichage des images
plt.figure(figsize=(12,4))

# Image originale
plt.subplot(1,3,1)
plt.imshow(image_originale, cmap='gray')
plt.title("Image originale")
plt.axis('off')

# Image tatouée
plt.subplot(1,3,2)
plt.imshow(image_tatouee, cmap='gray')
plt.title("Image tatouée")
plt.axis('off')

# Image attaquée
plt.subplot(1,3,3)
plt.imshow(image_attaquee, cmap='gray')
plt.title("Image attaquée")
plt.axis('off')

plt.tight_layout()
plt.show()




# Transformer en image 2D (carré si possible)
taille_cote = int(np.sqrt(len(watermark)))

wm_original_img = watermark[:taille_cote**2].reshape((taille_cote, taille_cote))
wm_extrait_img = watermark_extrait[:taille_cote**2].reshape((taille_cote, taille_cote))

plt.figure(figsize=(8,4))

# Watermark original
plt.subplot(1,2,1)
plt.imshow(wm_original_img, cmap='gray')
plt.title("Watermark original")
plt.axis('off')

# Watermark extrait
plt.subplot(1,2,2)
plt.imshow(wm_extrait_img, cmap='gray')
plt.title("Watermark extrait")
plt.axis('off')

plt.tight_layout()
plt.show()


#Afficher directement les erreurs :

diff_wm = wm_original_img != wm_extrait_img

plt.imshow(diff_wm, cmap='gray')
plt.title("Erreurs (blanc = erreur)")
plt.axis('off')
plt.show()



# 📌 Code pour afficher la différence  Différences
diff_tatouee = np.abs(image_originale - image_tatouee)
diff_attaquee = np.abs(image_originale - image_attaquee)



plt.figure(figsize=(10,4))

# Différence originale vs tatouée
plt.subplot(1,2,1)
plt.imshow(diff_tatouee, cmap='hot')
plt.title("Différence (Originale - Tatouée)")
plt.axis('off')

# Différence originale vs attaquée
plt.subplot(1,2,2)
plt.imshow(diff_attaquee, cmap='hot')
plt.title("Différence (Originale - Attaquée)")
plt.axis('off')

plt.tight_layout()
plt.show()
plt.imshow(diff_tatouee * 5, cmap='hot')