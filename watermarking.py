"""
Projet de tatouage d'image (Watermarking) avec DCT et QIM
Améliorations:
- Correction du bug de la méthode attaque_crop dupliquée
- Correction du bug temp.jpg hardcodé dans attaque_jpeg
- Ajout SSIM dans les métriques de l'API
- Meilleure gestion des erreurs
- Ajout d'une attaque "flou" (blur)
- Méthode evaluer() retourne aussi SSIM pour l'API
- Support images couleur (conversion auto en niveaux de gris)
"""

import cv2
import numpy as np
from scipy.fft import dctn, idctn
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import tempfile
import os
import warnings
warnings.filterwarnings('ignore')

# =========================================
# CLASSE PRINCIPALE
# =========================================

class WatermarkingDCT:
    """
    Classe pour l'insertion et l'extraction de watermark dans une image
    utilisant la DCT par blocs 8x8 et la modulation QIM
    """

    def __init__(self, image_path, block_size=8, delta=30, seed=42):
        """
        Initialise le système de watermarking

        Args:
            image_path (str): Chemin vers l'image
            block_size (int): Taille des blocs DCT (default: 8)
            delta (int): Pas de quantification QIM (default: 30)
            seed (int): Graine pour la génération aléatoire (default: 42)
        """
        # Chargement de l'image (support couleur -> niveaux de gris automatique)
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image non trouvée : {image_path}")

        if len(img.shape) == 3:
            self.image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        else:
            self.image = img.astype(np.float64)

        # Vérification des dimensions et recadrage si nécessaire
        h, w = self.image.shape
        if h % block_size != 0 or w % block_size != 0:
            new_h = h - (h % block_size)
            new_w = w - (w % block_size)
            print(f"⚠️ Redimensionnement de {h}x{w} à {new_h}x{new_w}")
            self.image = self.image[:new_h, :new_w]

        self.image_originale = self.image.copy()

        self.block_size = block_size
        self.delta = delta
        self.seed = seed
        self.watermark = None
        self.image_tatouee = None
        self.watermark_extrait = None

        # Calcul automatique de la taille du watermark
        h, w = self.image.shape
        self.nb_blocs = (h // block_size) * (w // block_size)

    def generer_watermark(self, taille=None):
        """
        Génère un watermark binaire aléatoire reproductible

        Args:
            taille (int): Taille du watermark (par défaut = nb_blocs)

        Returns:
            np.ndarray: Watermark binaire
        """
        if taille is None:
            taille = self.nb_blocs
        np.random.seed(self.seed)
        self.watermark = np.random.randint(0, 2, taille)
        return self.watermark

    def apply_dct_blocks(self, image):
        """Applique la DCT sur chaque bloc 8x8 de l'image"""
        h, w = image.shape
        block = self.block_size
        dct_image = np.zeros_like(image)

        for i in range(0, h, block):
            for j in range(0, w, block):
                bloc = image[i:i+block, j:j+block]
                dct_image[i:i+block, j:j+block] = dctn(bloc, norm='ortho')
        return dct_image

    def apply_idct_blocks(self, dct_image):
        """Applique la DCT inverse sur chaque bloc 8x8"""
        h, w = dct_image.shape
        block = self.block_size
        image_rec = np.zeros_like(dct_image)

        for i in range(0, h, block):
            for j in range(0, w, block):
                bloc = dct_image[i:i+block, j:j+block]
                image_rec[i:i+block, j:j+block] = idctn(bloc, norm='ortho')

        return np.clip(image_rec, 0, 255)

    def inserer_qim(self, dct_image, watermark, delta=None):
        """
        Insertion du watermark avec modulation QIM

        Args:
            dct_image (np.ndarray): Image en DCT
            watermark (np.ndarray): Watermark binaire à insérer
            delta (int): Pas de quantification

        Returns:
            np.ndarray: Image DCT modifiée
        """
        if delta is None:
            delta = self.delta

        h, w = dct_image.shape
        block = self.block_size
        dct_modif = dct_image.copy()
        idx = 0

        for i in range(0, h, block):
            for j in range(0, w, block):
                if idx >= len(watermark):
                    break
                coeff = dct_modif[i+3, j+4] if block == 8 else dct_modif[i+1, j+1]
                bit = watermark[idx]

                if bit == 0:
                    dct_modif[i+3, j+4] = delta * round(coeff / delta)
                else:
                    dct_modif[i+3, j+4] = delta * round(coeff / delta) + delta / 2
                idx += 1
        return dct_modif

    def extraire_qim(self, dct_image, taille_wm, delta=None):
        """
        Extraction du watermark depuis l'image DCT

        Args:
            dct_image (np.ndarray): Image en DCT
            taille_wm (int): Taille attendue du watermark
            delta (int): Pas de quantification

        Returns:
            np.ndarray: Watermark extrait
        """
        if delta is None:
            delta = self.delta

        h, w = dct_image.shape
        block = self.block_size
        watermark_extrait = []
        idx = 0

        for i in range(0, h, block):
            for j in range(0, w, block):
                if idx >= taille_wm:
                    break
                coeff = dct_image[i+3, j+4] if block == 8 else dct_image[i+1, j+1]
                q = round(coeff / (delta / 2))
                bit = q % 2
                watermark_extrait.append(bit)
                idx += 1

        return np.array(watermark_extrait)

    def inserer(self):
        """Pipeline complet d'insertion du watermark"""
        if self.watermark is None:
            self.generer_watermark()

        dct_image = self.apply_dct_blocks(self.image)
        dct_modif = self.inserer_qim(dct_image, self.watermark)
        self.image_tatouee = self.apply_idct_blocks(dct_modif)

        return self.image_tatouee

    def extraire(self, image, delta=None):
        """
        Extraction du watermark depuis une image

        Args:
            image (np.ndarray): Image contenant le watermark
            delta (int): Pas de quantification

        Returns:
            np.ndarray: Watermark extrait
        """
        dct_image = self.apply_dct_blocks(image)
        self.watermark_extrait = self.extraire_qim(dct_image, self.nb_blocs, delta)
        return self.watermark_extrait

    # =========================================
    # ATTAQUES
    # =========================================

    @staticmethod
    def attaque_bruit(image, sigma=10):
        """Ajout de bruit gaussien"""
        bruit = np.random.normal(0, sigma, image.shape)
        return np.clip(image + bruit, 0, 255)

    @staticmethod
    def attaque_jpeg(image, qualite=50):
        """Compression JPEG — utilise un fichier temporaire sécurisé"""
        # CORRECTION: utilisation de tempfile au lieu de "temp.jpg" hardcodé
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            cv2.imwrite(tmp_path, image.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, qualite])
            result = cv2.imread(tmp_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
        finally:
            os.unlink(tmp_path)
        return result

    @staticmethod
    def attaque_crop(image, pourcentage=0.9):
        """Recadrage + redimensionnement à la taille originale"""
        # CORRECTION: suppression du doublon (méthode était définie deux fois)
        h, w = image.shape
        new_h, new_w = int(h * pourcentage), int(w * pourcentage)
        image_crop = image[:new_h, :new_w]
        return cv2.resize(image_crop, (w, h))

    @staticmethod
    def attaque_rotation(image, angle=5):
        """Rotation légère"""
        h, w = image.shape
        centre = (w // 2, h // 2)
        matrice = cv2.getRotationMatrix2D(centre, angle, 1.0)
        return cv2.warpAffine(image, matrice, (w, h))

    @staticmethod
    def attaque_flou(image, ksize=5):
        """
        Attaque par flou gaussien (NOUVEAU)

        Args:
            image (np.ndarray): Image en niveaux de gris
            ksize (int): Taille du noyau (doit être impair)

        Returns:
            np.ndarray: Image floutée
        """
        if ksize % 2 == 0:
            ksize += 1  # Forcer impair
        return cv2.GaussianBlur(image.astype(np.uint8), (ksize, ksize), 0).astype(np.float64)

    # =========================================
    # MÉTRIQUES D'ÉVALUATION
    # =========================================

    @staticmethod
    def calculer_ber(wm_original, wm_extrait):
        """Bit Error Rate"""
        min_len = min(len(wm_original), len(wm_extrait))
        erreurs = np.sum(wm_original[:min_len] != wm_extrait[:min_len])
        return float(erreurs / min_len) if min_len > 0 else 1.0

    @staticmethod
    def calculer_accuracy(wm_original, wm_extrait):
        """Accuracy du watermark extrait"""
        min_len = min(len(wm_original), len(wm_extrait))
        correct = np.sum(wm_original[:min_len] == wm_extrait[:min_len])
        return float(correct / min_len) if min_len > 0 else 0.0

    def evaluer(self, image_test, description=""):
        """
        Évalue la qualité de l'image et l'extraction du watermark

        Args:
            image_test (np.ndarray): Image à évaluer
            description (str): Description pour l'affichage

        Returns:
            dict: Métriques PSNR, SSIM, BER, Accuracy
        """
        wm_extrait = self.extraire(image_test)

        psnr_val = psnr(self.image_originale, image_test, data_range=255)
        ssim_val = ssim(self.image_originale, image_test, data_range=255)
        ber_val = self.calculer_ber(self.watermark, wm_extrait)
        acc_val = self.calculer_accuracy(self.watermark, wm_extrait)

        if description:
            print(f"\n--- {description} ---")
            print(f"PSNR      : {psnr_val:.2f} dB")
            print(f"SSIM      : {ssim_val:.4f}")
            print(f"BER       : {ber_val:.4f}")
            print(f"Accuracy  : {acc_val:.4f}")

        return {
            'description': description,
            'PSNR': psnr_val,
            'SSIM': ssim_val,
            'BER': ber_val,
            'Accuracy': acc_val,
            'watermark_extrait': wm_extrait
        }


# =========================================
# FONCTIONS DE VISUALISATION
# =========================================

def afficher_comparaison(watermarking, image_attaquee=None, type_attaque="Aucune"):
    """Affiche toutes les visualisations du projet"""
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(watermarking.image_originale, cmap='gray')
    plt.title("Image originale")
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(watermarking.image_tatouee, cmap='gray')
    psnr_val = psnr(watermarking.image_originale, watermarking.image_tatouee, data_range=255)
    plt.title(f"Image tatouée\nPSNR: {psnr_val:.2f} dB")
    plt.axis('off')

    if image_attaquee is not None:
        plt.subplot(2, 3, 3)
        plt.imshow(image_attaquee, cmap='gray')
        plt.title(f"Image attaquée\n{type_attaque}")
        plt.axis('off')

    taille_cote = int(np.sqrt(watermarking.nb_blocs))
    if taille_cote * taille_cote > 0:
        wm_original_img = watermarking.watermark[:taille_cote**2].reshape((taille_cote, taille_cote))

        if watermarking.watermark_extrait is not None:
            wm_extrait_img = watermarking.watermark_extrait[:taille_cote**2].reshape((taille_cote, taille_cote))

            plt.subplot(2, 3, 4)
            plt.imshow(wm_original_img, cmap='gray')
            plt.title("Watermark original")
            plt.axis('off')

            plt.subplot(2, 3, 5)
            plt.imshow(wm_extrait_img, cmap='gray')
            plt.title("Watermark extrait")
            plt.axis('off')

            diff_wm = wm_original_img != wm_extrait_img
            plt.subplot(2, 3, 6)
            plt.imshow(diff_wm, cmap='gray')
            plt.title(f"Erreurs (blanc)\n{np.sum(diff_wm)} erreurs")
            plt.axis('off')

    plt.tight_layout()
    plt.show()


def tester_deltas(watermarking, deltas=[10, 20, 30, 40, 50], qualite_jpeg=70):
    """Teste différentes valeurs de delta"""
    print("\n" + "="*60)
    print("TEST DE DIFFÉRENTES VALEURS DE DELTA")
    print("="*60)

    resultats = []

    for delta in deltas:
        dct_image = watermarking.apply_dct_blocks(watermarking.image_originale)
        dct_modif = watermarking.inserer_qim(dct_image, watermarking.watermark, delta=delta)
        image_tatouee = watermarking.apply_idct_blocks(dct_modif)

        psnr_sans = psnr(watermarking.image_originale, image_tatouee, data_range=255)

        image_attaquee = watermarking.attaque_jpeg(image_tatouee, qualite=qualite_jpeg)
        dct_attaquee = watermarking.apply_dct_blocks(image_attaquee)
        wm_extrait = watermarking.extraire_qim(dct_attaquee, watermarking.nb_blocs, delta=delta)
        ber = watermarking.calculer_ber(watermarking.watermark, wm_extrait)

        resultats.append({'delta': delta, 'PSNR': psnr_sans, 'BER': ber})
        print(f"delta={delta:2d} → PSNR={psnr_sans:.2f} dB, BER={ber:.4f}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot([r['delta'] for r in resultats], [r['PSNR'] for r in resultats], 'b-o')
    plt.xlabel('Delta')
    plt.ylabel('PSNR (dB)')
    plt.title('Impact du delta sur la qualité image')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot([r['delta'] for r in resultats], [r['BER'] for r in resultats], 'r-o')
    plt.xlabel('Delta')
    plt.ylabel('BER')
    plt.title('Impact du delta sur la robustesse')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return resultats


# =========================================
# PROGRAMME PRINCIPAL
# =========================================

def main():
    print("="*60)
    print("PROJET DE WATERMARKING DCT + QIM")
    print("="*60)

    chemin_image = "moun.jpg"

    try:
        watermarking = WatermarkingDCT(chemin_image, delta=30, seed=42)
        print(f"✅ Image chargée : {watermarking.image.shape}")
        print(f"   Nombre de blocs : {watermarking.nb_blocs}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    watermarking.generer_watermark()
    print(f"✅ Watermark généré : {watermarking.watermark.shape[0]} bits")

    print("\n--- INSERTION ---")
    watermarking.inserer()
    watermarking.evaluer(watermarking.image_tatouee, "SANS ATTAQUE")

    print("\n" + "-"*40)
    print("Choisissez une attaque :")
    print("  1. Bruit gaussien")
    print("  2. Compression JPEG")
    print("  3. Recadrage")
    print("  4. Rotation")
    print("  5. Flou gaussien")
    print("  6. Tester plusieurs deltas")
    print("  7. Aucune attaque")

    choix = input("\nVotre choix (1-7) : ").strip()

    image_attaquee = watermarking.image_tatouee.copy()
    type_attaque = "Aucune"

    if choix == "1":
        sigma = float(input("Sigma du bruit (default 10) : ") or "10")
        image_attaquee = watermarking.attaque_bruit(watermarking.image_tatouee, sigma=sigma)
        type_attaque = f"Bruit gaussien (σ={sigma})"
    elif choix == "2":
        qualite = int(input("Qualité JPEG (1-100, default 70) : ") or "70")
        image_attaquee = watermarking.attaque_jpeg(watermarking.image_tatouee, qualite=qualite)
        type_attaque = f"JPEG (qualité={qualite})"
    elif choix == "3":
        pourcentage = float(input("Pourcentage de conservation (0.5-1.0, default 0.9) : ") or "0.9")
        image_attaquee = watermarking.attaque_crop(watermarking.image_tatouee, pourcentage=pourcentage)
        type_attaque = f"Recadrage ({pourcentage*100:.0f}%)"
    elif choix == "4":
        angle = float(input("Angle de rotation (deg, default 5) : ") or "5")
        image_attaquee = watermarking.attaque_rotation(watermarking.image_tatouee, angle=angle)
        type_attaque = f"Rotation ({angle}°)"
    elif choix == "5":
        ksize = int(input("Taille noyau flou (impair, default 5) : ") or "5")
        image_attaquee = watermarking.attaque_flou(watermarking.image_tatouee, ksize=ksize)
        type_attaque = f"Flou gaussien ({ksize}x{ksize})"
    elif choix == "6":
        tester_deltas(watermarking, [10, 20, 30, 40, 50, 60])
        image_attaquee = watermarking.image_tatouee
        type_attaque = "Aucune (test deltas effectué)"
    else:
        image_attaquee = watermarking.image_tatouee
        type_attaque = "Aucune"

    if choix not in ["6", "7"]:
        print("\n--- EXTRACTION APRÈS ATTAQUE ---")
        watermarking.extraire(image_attaquee)
        watermarking.evaluer(image_attaquee, f"APRÈS ATTAQUE : {type_attaque}")

    print("\n--- VISUALISATION ---")
    afficher_comparaison(watermarking, image_attaquee if choix not in ["6", "7"] else None, type_attaque)

    print("\n✅ Programme terminé !")


if __name__ == "__main__":
    main()
