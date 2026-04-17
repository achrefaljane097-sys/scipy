import cv2
import numpy as np
from scipy.fft import dctn, idctn
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

# =========================================
# ETAPE 1 : CHARGER L'IMAGE
# =========================================
image_originale = cv2.imread("moun.jpg", cv2.IMREAD_GRAYSCALE)
if image_originale is None:                          # ✅ AJOUT : vérification comme ton code
    print("Erreur : image non trouvée")
    exit()
image_originale = image_originale.astype(np.float64)

plt.imshow(image_originale, cmap='gray')             # ✅ AJOUT : affichage immédiat (ton code)
plt.title("Image originale")
plt.axis('off')
plt.show()

# =========================================
# ETAPE 2 : DCT PAR BLOCS 8x8
# =========================================
def apply_dct_blocks(image):
    h, w = image.shape
    dct_image = np.zeros_like(image)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            bloc = image[i:i+8, j:j+8]
            dct_image[i:i+8, j:j+8] = dctn(bloc, norm='ortho')
    return dct_image

def appliquer_idct_blocks(dct_image):
    h, w = dct_image.shape
    image_rec = np.zeros_like(dct_image)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            bloc = dct_image[i:i+8, j:j+8]
            image_rec[i:i+8, j:j+8] = idctn(bloc, norm='ortho')
    return np.clip(image_rec, 0, 255)

# =========================================
# ETAPE 3 : VISUALISATION DCT
# =========================================
dct_image = apply_dct_blocks(image_originale)

plt.imshow(np.log(np.abs(dct_image) + 1), cmap='gray')  # ✅ AJOUT : visualisation DCT (ton code)
plt.title("DCT (visualisation log)")
plt.colorbar()
plt.show()

# =========================================
# ETAPE 4 : WATERMARK
# =========================================
def generer_watermark(taille, cle=42):
    np.random.seed(cle)
    return np.random.randint(0, 2, taille)

taille = (image_originale.shape[0] // 8) * (image_originale.shape[1] // 8)
watermark = generer_watermark(taille)
print("Watermark généré :", watermark[:20], "... (affiché les 20 premiers)")  # ✅ AJOUT : print (ton code)

# =========================================
# ETAPE 5 : INSERTION QIM AMÉLIORÉE
# =========================================
# ✅ AJOUT : formule QIM avec décalage delta/2 (ton code, variante supérieure)
# Le code original utilisait parité paire/impaire
# La nouvelle formule crée deux réseaux décalés de delta/2 → meilleure séparation des états
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
            if bit == 0:
                dct_modif[i+3, j+4] = delta * round(coeff / delta)
            else:
                dct_modif[i+3, j+4] = delta * round(coeff / delta) + delta / 2
            idx += 1
    return dct_modif

# ✅ AJOUT : formule extraction cohérente avec la nouvelle insertion
def extraire_qim(dct_image, taille_wm, delta=30):
    h, w = dct_image.shape
    watermark_extrait = []
    idx = 0
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            if idx >= taille_wm:
                break
            coeff = dct_image[i+3, j+4]
            bit = int(round(coeff / (delta / 2))) % 2
            watermark_extrait.append(abs(bit))
            idx += 1
    return np.array(watermark_extrait)

# =========================================
# ETAPE 6 : ATTAQUES
# =========================================
def attaque_bruit(image, sigma=10):
    bruit = np.random.normal(0, sigma, image.shape)
    return np.clip(image + bruit, 0, 255)

def attaque_jpeg(image, qualite=50):
    cv2.imwrite("temp.jpg", image.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, qualite])
    return cv2.imread("temp.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float64)

# =========================================
# ETAPE 7 : CALCUL BER
# =========================================
def calculer_ber(wm_original, wm_extrait):
    erreurs = np.sum(wm_original != wm_extrait)
    return erreurs / len(wm_original)

# =========================================
# ETAPE 8 : INSERTION ET MESURE SANS ATTAQUE
# =========================================
# Insertion
dct_modif = inserer_qim(dct_image, watermark)
image_tatouee = appliquer_idct_blocks(dct_modif)

# Extraction sans attaque
dct_tatouee = apply_dct_blocks(image_tatouee)
watermark_extrait_sans = extraire_qim(dct_tatouee, len(watermark))

# ✅ AJOUT : mesure PSNR et BER AVANT attaque (ton code)
# Permet de séparer l'impact de l'insertion vs l'impact de l'attaque
psnr_sans = psnr(image_originale, image_tatouee, data_range=255)
ber_sans   = calculer_ber(watermark, watermark_extrait_sans)
print(f"\n--- Sans attaque ---")
print(f"PSNR  : {psnr_sans:.2f} dB")
print(f"BER   : {ber_sans:.4f}")

# =========================================
# ETAPE 9 : ATTAQUE ET MESURE APRES ATTAQUE
# =========================================
# ✅ AJOUT : choix interactif de l'attaque (ton code)
choix = input("\nChoisir attaque → (1) Bruit gaussien  (2) JPEG  (3) Aucune : ")

if choix == "1":
    image_attaquee = attaque_bruit(image_tatouee)
elif choix == "2":
    image_attaquee = attaque_jpeg(image_tatouee)      # code original uniquement
else:
    image_attaquee = image_tatouee.copy()

# Extraction après attaque
dct_attaquee = apply_dct_blocks(image_attaquee)
watermark_extrait_avec = extraire_qim(dct_attaquee, len(watermark))

psnr_avec = psnr(image_originale, image_attaquee, data_range=255)
ber_avec   = calculer_ber(watermark, watermark_extrait_avec)
print(f"\n--- Après attaque ---")
print(f"PSNR  : {psnr_avec:.2f} dB")
print(f"BER   : {ber_avec:.4f}")

# =========================================
# ETAPE 10 : AFFICHAGE COMPARAISON IMAGES
# =========================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image_originale, cmap='gray')
plt.title("Image originale")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(image_tatouee, cmap='gray')
plt.title("Image tatouée")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(image_attaquee, cmap='gray')
plt.title("Image attaquée")
plt.axis('off')

plt.tight_layout()
plt.show()

# =========================================
# ETAPE 11 : AFFICHAGE WATERMARK ORIGINAL VS EXTRAIT
# =========================================
taille_cote = int(np.sqrt(len(watermark)))

wm_original_img = watermark[:taille_cote**2].reshape((taille_cote, taille_cote))
wm_extrait_img  = watermark_extrait_avec[:taille_cote**2].reshape((taille_cote, taille_cote))

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(wm_original_img, cmap='gray')
plt.title("Watermark original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(wm_extrait_img, cmap='gray')
plt.title("Watermark extrait")
plt.axis('off')

plt.tight_layout()
plt.show()

# =========================================
# ETAPE 12 : CARTE DES ERREURS
# =========================================
diff_wm = wm_original_img != wm_extrait_img

plt.imshow(diff_wm, cmap='gray')
plt.title("Erreurs (blanc = erreur)")
plt.axis('off')
plt.show()

# =========================================
# ETAPE 13 : CARTES DE DIFFERENCE (AMPLIFIEES)
# =========================================
diff_tatouee  = np.abs(image_originale - image_tatouee)
diff_attaquee = np.abs(image_originale - image_attaquee)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(diff_tatouee, cmap='hot')
plt.title("Différence (Originale - Tatouée)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(diff_attaquee, cmap='hot')
plt.title("Différence (Originale - Attaquée)")
plt.axis('off')

plt.tight_layout()
plt.show()

plt.imshow(diff_tatouee * 5, cmap='hot')
plt.title("Différence tatouée × 5 (amplifiée)")
plt.axis('off')
plt.show()
