"""
Watermarking DCT + QIM — Version 2.0
Nouvelles attaques : contraste, filtre médian, sel & poivre
"""

import cv2
import numpy as np
from scipy.fft import dctn, idctn
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import tempfile
import os
import warnings
warnings.filterwarnings('ignore')


class WatermarkingDCT:
    def __init__(self, image_path, block_size=8, delta=30, seed=42):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image non trouvée : {image_path}")
        if len(img.shape) == 3:
            self.image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        else:
            self.image = img.astype(np.float64)

        h, w = self.image.shape
        if h % block_size != 0 or w % block_size != 0:
            self.image = self.image[:h - h % block_size, :w - w % block_size]

        self.image_originale = self.image.copy()
        self.block_size = block_size
        self.delta = delta
        self.seed = seed
        self.watermark = None
        self.image_tatouee = None
        self.watermark_extrait = None
        h, w = self.image.shape
        self.nb_blocs = (h // block_size) * (w // block_size)

    def generer_watermark(self, taille=None):
        if taille is None:
            taille = self.nb_blocs
        np.random.seed(self.seed)
        self.watermark = np.random.randint(0, 2, taille)
        return self.watermark

    def apply_dct_blocks(self, image):
        h, w = image.shape
        b = self.block_size
        out = np.zeros_like(image)
        for i in range(0, h, b):
            for j in range(0, w, b):
                out[i:i+b, j:j+b] = dctn(image[i:i+b, j:j+b], norm='ortho')
        return out

    def apply_idct_blocks(self, dct_image):
        h, w = dct_image.shape
        b = self.block_size
        out = np.zeros_like(dct_image)
        for i in range(0, h, b):
            for j in range(0, w, b):
                out[i:i+b, j:j+b] = idctn(dct_image[i:i+b, j:j+b], norm='ortho')
        return np.clip(out, 0, 255)

    def inserer_qim(self, dct_image, watermark, delta=None):
        if delta is None:
            delta = self.delta
        h, w = dct_image.shape
        b = self.block_size
        out = dct_image.copy()
        idx = 0
        for i in range(0, h, b):
            for j in range(0, w, b):
                if idx >= len(watermark):
                    break
                coeff = out[i+3, j+4]
                bit = watermark[idx]
                if bit == 0:
                    out[i+3, j+4] = delta * round(coeff / delta)
                else:
                    out[i+3, j+4] = delta * round(coeff / delta) + delta / 2
                idx += 1
        return out

    def extraire_qim(self, dct_image, taille_wm, delta=None):
        if delta is None:
            delta = self.delta
        h, w = dct_image.shape
        b = self.block_size
        wm = []
        idx = 0
        for i in range(0, h, b):
            for j in range(0, w, b):
                if idx >= taille_wm:
                    break
                coeff = dct_image[i+3, j+4]
                q = round(coeff / (delta / 2))
                wm.append(q % 2)
                idx += 1
        return np.array(wm)

    def inserer(self):
        if self.watermark is None:
            self.generer_watermark()
        dct = self.apply_dct_blocks(self.image)
        dct_mod = self.inserer_qim(dct, self.watermark)
        self.image_tatouee = self.apply_idct_blocks(dct_mod)
        return self.image_tatouee

    def extraire(self, image, delta=None):
        dct = self.apply_dct_blocks(image)
        self.watermark_extrait = self.extraire_qim(dct, self.nb_blocs, delta)
        return self.watermark_extrait

    # ── ATTAQUES CLASSIQUES ────────────────────────────────────────────
    @staticmethod
    def attaque_bruit(image, sigma=10):
        bruit = np.random.normal(0, sigma, image.shape)
        return np.clip(image + bruit, 0, 255)

    @staticmethod
    def attaque_jpeg(image, qualite=50):
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
        h, w = image.shape
        new_h, new_w = int(h * pourcentage), int(w * pourcentage)
        return cv2.resize(image[:new_h, :new_w], (w, h))

    @staticmethod
    def attaque_rotation(image, angle=5):
        h, w = image.shape
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))

    @staticmethod
    def attaque_flou(image, ksize=5):
        if ksize % 2 == 0: ksize += 1
        return cv2.GaussianBlur(image.astype(np.uint8), (ksize, ksize), 0).astype(np.float64)

    # ── NOUVELLES ATTAQUES ─────────────────────────────────────────────
    @staticmethod
    def attaque_contraste(image, alpha=1.5):
        """
        Ajustement de contraste (NOUVEAU)
        alpha > 1 : augmente le contraste
        alpha < 1 : réduit le contraste
        """
        return np.clip(image * alpha, 0, 255)

    @staticmethod
    def attaque_median(image, ksize=3):
        """
        Filtre médian (NOUVEAU) — efficace contre le bruit impulsionnel
        Préserve les bords mais modifie les coefficients DCT
        """
        if ksize % 2 == 0: ksize += 1
        return cv2.medianBlur(image.astype(np.uint8), ksize).astype(np.float64)

    @staticmethod
    def attaque_sel_poivre(image, prob=0.05):
        """
        Bruit sel & poivre (NOUVEAU)
        Remplace aléatoirement des pixels par 0 (poivre) ou 255 (sel)
        """
        result = image.copy()
        mask = np.random.random(image.shape)
        result[mask < prob / 2] = 0       # poivre
        result[mask > 1 - prob / 2] = 255  # sel
        return result

    # ── MÉTRIQUES ──────────────────────────────────────────────────────
    @staticmethod
    def calculer_ber(wm_original, wm_extrait):
        n = min(len(wm_original), len(wm_extrait))
        return float(np.sum(wm_original[:n] != wm_extrait[:n]) / n) if n > 0 else 1.0

    @staticmethod
    def calculer_accuracy(wm_original, wm_extrait):
        n = min(len(wm_original), len(wm_extrait))
        return float(np.sum(wm_original[:n] == wm_extrait[:n]) / n) if n > 0 else 0.0

    def evaluer(self, image_test, description=""):
        wm_ext = self.extraire(image_test)
        psnr_val = psnr(self.image_originale, image_test, data_range=255)
        ssim_val = ssim(self.image_originale, image_test, data_range=255)
        ber_val = self.calculer_ber(self.watermark, wm_ext)
        acc_val = self.calculer_accuracy(self.watermark, wm_ext)
        return {
            'description': description,
            'PSNR': psnr_val,
            'SSIM': ssim_val,
            'BER': ber_val,
            'Accuracy': acc_val,
            'watermark_extrait': wm_ext
        }
