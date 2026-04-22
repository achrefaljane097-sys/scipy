"""
Projet de tatouage d'image (Watermarking) avec DCT et QIM
Auteur: achrefaljane097
Fonctionnalités:
- Insertion de watermark binaire dans l'image
- Extraction du watermark
- Attaques (bruit gaussien, JPEG)
- Métriques d'évaluation (PSNR, BER, SSIM, Accuracy)
- Tests avec différentes valeurs de delta
- Visualisation complète des résultats
"""

import cv2
import numpy as np
from scipy.fft import dctn, idctn
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
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
        # Chargement de l'image
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise FileNotFoundError(f"Image non trouvée : {image_path}")
        
        self.image = self.image.astype(np.float64)
        self.image_originale = self.image.copy()
        
        # Vérification des dimensions
        h, w = self.image.shape
        if h % block_size != 0 or w % block_size != 0:
            new_h = h - (h % block_size)
            new_w = w - (w % block_size)
            print(f"⚠️ Redimensionnement de {h}x{w} à {new_h}x{new_w}")
            self.image = self.image[:new_h, :new_w]

        self.image_originale = self.image.copy()  # ← AJOUTER CETTE LIGNE

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
        Génère un watermark binaire aléatoire
        
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
        """
        Applique la DCT sur chaque bloc 8x8 de l'image
        
        Args:
            image (np.ndarray): Image d'entrée
        
        Returns:
            np.ndarray: Image transformée en DCT
        """
        h, w = image.shape
        block = self.block_size
        dct_image = np.zeros_like(image)
        
        for i in range(0, h, block):
            for j in range(0, w, block):
                bloc = image[i:i+block, j:j+block]
                dct_image[i:i+block, j:j+block] = dctn(bloc, norm='ortho')
        return dct_image
    
    def apply_idct_blocks(self, dct_image):
        """
        Applique la DCT inverse sur chaque bloc 8x8
        
        Args:
            dct_image (np.ndarray): Image en DCT
        
        Returns:
            np.ndarray: Image reconstruite
        """
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
        Insertion du watermark avec modulation QIM améliorée
        
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
                # Utilisation du coefficient DC (hautes fréquences pour robustesse)
                coeff = dct_modif[i+3, j+4] if block == 8 else dct_modif[i+1, j+1]
                bit = watermark[idx]
                
                # Quantification QIM avec décalage delta/2
                if bit == 0:
                    dct_modif[i+3, j+4] = delta * round(coeff / delta)
                else:
                    dct_modif[i+3, j+4] = delta * round(coeff / delta) + delta/2
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
                # Décision binaire basée sur l'arrondi au multiple de delta/2 le plus proche
                q = round(coeff / (delta/2))
                bit = q % 2
                watermark_extrait.append(bit)
                idx += 1
        
        return np.array(watermark_extrait)
    
    def inserer(self):
        """
        Pipeline complet d'insertion du watermark
        """
        # Génération du watermark si nécessaire
        if self.watermark is None:
            self.generer_watermark()
        
        # DCT de l'image originale
        dct_image = self.apply_dct_blocks(self.image)
        
        # Insertion
        dct_modif = self.inserer_qim(dct_image, self.watermark)
        
        # IDCT pour reconstruire l'image tatouée
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
        """Compression JPEG"""
        cv2.imwrite("temp.jpg", image.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, qualite])
        return cv2.imread("temp.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float64)
    
    @staticmethod
    def attaque_crop(image, pourcentage=0.9):
        """Recadrage"""
        h, w = image.shape
        new_h, new_w = int(h * pourcentage), int(w * pourcentage)
        return image[:new_h, :new_w]
    
    @staticmethod
    def attaque_crop(image, pourcentage=0.9):
        """Recadrage de l'image"""
        h, w = image.shape
        new_h, new_w = int(h * pourcentage), int(w * pourcentage)
    # Recadrer
        image_crop = image[:new_h, :new_w]
    # Re-dimensionner à la taille originale
        return cv2.resize(image_crop, (w, h))
    @staticmethod
    def attaque_rotation(image, angle=5):
        """Rotation légère"""
        h, w = image.shape
        centre = (w//2, h//2)
        matrice = cv2.getRotationMatrix2D(centre, angle, 1.0)
        return cv2.warpAffine(image, matrice, (w, h))
    
    # =========================================
    # MÉTRIQUES D'ÉVALUATION
    # =========================================
    
    @staticmethod
    def calculer_ber(wm_original, wm_extrait):
        """Bit Error Rate"""
        min_len = min(len(wm_original), len(wm_extrait))
        erreurs = np.sum(wm_original[:min_len] != wm_extrait[:min_len])
        return erreurs / min_len if min_len > 0 else 1.0
    
    @staticmethod
    def calculer_accuracy(wm_original, wm_extrait):
        """Accuracy du watermark extrait"""
        min_len = min(len(wm_original), len(wm_extrait))
        correct = np.sum(wm_original[:min_len] == wm_extrait[:min_len])
        return correct / min_len if min_len > 0 else 0.0
    
    def evaluer(self, image_test, description=""):
        """
        Évalue la qualité de l'image et l'extraction du watermark
        
        Args:
            image_test (np.ndarray): Image à évaluer
            description (str): Description pour l'affichage
        
        Returns:
            dict: Dictionnaire des métriques
        """
        # Extraction
        wm_extrait = self.extraire(image_test)
        
        # Métriques
        psnr_val = psnr(self.image_originale, image_test, data_range=255)
        ssim_val = ssim(self.image_originale, image_test, data_range=255)
        ber_val = self.calculer_ber(self.watermark, wm_extrait)
        acc_val = self.calculer_accuracy(self.watermark, wm_extrait)
        
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
    """
    Affiche toutes les visualisations du projet
    """
    plt.figure(figsize=(15, 10))
    
    # Image originale
    plt.subplot(2, 3, 1)
    plt.imshow(watermarking.image_originale, cmap='gray')
    plt.title("Image originale")
    plt.axis('off')
    
    # Image tatouée
    plt.subplot(2, 3, 2)
    plt.imshow(watermarking.image_tatouee, cmap='gray')
    plt.title(f"Image tatouée\nPSNR: {psnr(watermarking.image_originale, watermarking.image_tatouee, data_range=255):.2f} dB")
    plt.axis('off')
    
    # Image attaquée (si fournie)
    if image_attaquee is not None:
        plt.subplot(2, 3, 3)
        plt.imshow(image_attaquee, cmap='gray')
        plt.title(f"Image attaquée\n{type_attaque}")
        plt.axis('off')
    
    # Watermark original vs extrait
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
            
            # Carte des erreurs
            diff_wm = wm_original_img != wm_extrait_img
            plt.subplot(2, 3, 6)
            plt.imshow(diff_wm, cmap='gray')
            plt.title(f"Erreurs (blanc)\n{np.sum(diff_wm)} erreurs")
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def afficher_differences(watermarking, image_attaquee):
    """
    Affiche les différences entre les images
    """
    diff_tatouee = np.abs(watermarking.image_originale - watermarking.image_tatouee)
    diff_attaquee = np.abs(watermarking.image_originale - image_attaquee)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(diff_tatouee, cmap='hot')
    plt.title("Différence (Originale - Tatouée)")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(diff_attaquee, cmap='hot')
    plt.title("Différence (Originale - Attaquée)")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(diff_tatouee * 5, cmap='hot')
    plt.title("Différence Tatouée × 5 (amplifiée)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def tester_deltas(watermarking, deltas=[10, 20, 30, 40, 50], qualite_jpeg=70):
    """
    Teste différentes valeurs de delta pour trouver le meilleur compromis
    """
    print("\n" + "="*60)
    print("TEST DE DIFFÉRENTES VALEURS DE DELTA")
    print("="*60)
    
    resultats = []
    
    for delta in deltas:
        # Insertion avec ce delta
        dct_image = watermarking.apply_dct_blocks(watermarking.image_originale)
        dct_modif = watermarking.inserer_qim(dct_image, watermarking.watermark, delta=delta)
        image_tatouee = watermarking.apply_idct_blocks(dct_modif)
        
        # PSNR sans attaque
        psnr_sans = psnr(watermarking.image_originale, image_tatouee, data_range=255)
        
        # Après attaque JPEG
        image_attaquee = watermarking.attaque_jpeg(image_tatouee, qualite=qualite_jpeg)
        dct_attaquee = watermarking.apply_dct_blocks(image_attaquee)
        wm_extrait = watermarking.extraire_qim(dct_attaquee, watermarking.nb_blocs, delta=delta)
        ber = watermarking.calculer_ber(watermarking.watermark, wm_extrait)
        
        resultats.append({
            'delta': delta,
            'PSNR': psnr_sans,
            'BER': ber
        })
        
        print(f"delta={delta:2d} → PSNR={psnr_sans:.2f} dB, BER={ber:.4f}")
    
    # Graphique des résultats
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
    
    # =========================================
    # 1. INITIALISATION
    # =========================================
    
    chemin_image = "moun.jpg"
    
    try:
        watermarking = WatermarkingDCT(chemin_image, delta=30, seed=42)
        print(f"✅ Image chargée : {watermarking.image.shape}")
        print(f"   Nombre de blocs : {watermarking.nb_blocs}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("   Assurez-vous que 'moun.jpg' est dans le même dossier")
        return
    
    # Génération du watermark
    watermarking.generer_watermark()
    print(f"✅ Watermark généré : {watermarking.watermark.shape[0]} bits")
    
    # =========================================
    # 2. INSERTION
    # =========================================
    
    print("\n--- INSERTION ---")
    watermarking.inserer()
    
    # Évaluation sans attaque
    watermarking.evaluer(watermarking.image_tatouee, "SANS ATTAQUE")
    
    # =========================================
    # 3. MENU DES ATTAQUES
    # =========================================
    
    print("\n" + "-"*40)
    print("Choisissez une attaque à appliquer :")
    print("  1. Bruit gaussien")
    print("  2. Compression JPEG")
    print("  3. Recadrage")
    print("  4. Rotation")
    print("  5. Tester plusieurs deltas")
    print("  6. Aucune attaque (juste visualiser)")
    
    choix = input("\nVotre choix (1-6) : ").strip()
    
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
        type_attaque = f"Recadrage ({pourcentage*100}%)"
        
    elif choix == "4":
        angle = float(input("Angle de rotation (deg, default 5) : ") or "5")
        image_attaquee = watermarking.attaque_rotation(watermarking.image_tatouee, angle=angle)
        type_attaque = f"Rotation ({angle}°)"
        
    elif choix == "5":
        deltas = [10, 20, 30, 40, 50, 60]
        tester_deltas(watermarking, deltas)
        # On continue avec delta=30 pour la suite
        image_attaquee = watermarking.image_tatouee
        type_attaque = "Aucune (test deltas effectué)"
        
    else:
        image_attaquee = watermarking.image_tatouee
        type_attaque = "Aucune"
    
    # =========================================
    # 4. EXTRACTION APRÈS ATTAQUE (si applicable)
    # =========================================
    
    if choix != "5" and choix != "6":
        print("\n--- EXTRACTION APRÈS ATTAQUE ---")
        watermarking.extraire(image_attaquee)
        watermarking.evaluer(image_attaquee, f"APRÈS ATTAQUE : {type_attaque}")
    
    # =========================================
    # 5. VISUALISATIONS
    # =========================================
    
    print("\n--- VISUALISATION ---")
    afficher_comparaison(watermarking, image_attaquee if choix not in ["5", "6"] else None, type_attaque)
    
    if choix not in ["5", "6"]:
        afficher_differences(watermarking, image_attaquee)
    
    print("\n✅ Programme terminé !")


# =========================================
# EXÉCUTION
# =========================================

if __name__ == "__main__":
    main()
