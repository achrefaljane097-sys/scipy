from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import os
import threading
from watermarking import WatermarkingDCT

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Lock pour la sécurité des threads
_lock = threading.Lock()
watermarking = None


def image_to_b64(image_array: np.ndarray) -> str:
    """Convertit un tableau numpy en base64 JPEG."""
    success, buffer = cv2.imencode('.jpg', image_array.astype(np.uint8),
                                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not success:
        raise ValueError("Échec de l'encodage de l'image")
    return base64.b64encode(buffer).decode()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    global watermarking

    if 'image' not in request.files:
        return jsonify({'error': 'Aucune image fournie'}), 400

    file = request.files['image']
    if not file or file.filename == '':
        return jsonify({'error': 'Fichier invalide'}), 400

    # Vérification du type MIME
    allowed_types = {'image/jpeg', 'image/png', 'image/bmp', 'image/tiff'}
    if file.content_type not in allowed_types:
        return jsonify({'error': 'Format non supporté. Utilisez JPEG, PNG, BMP ou TIFF'}), 400

    path = os.path.join(UPLOAD_FOLDER, 'upload.jpg')

    try:
        file.save(path)

        with _lock:
            watermarking = WatermarkingDCT(path, delta=30, seed=42)
            watermarking.generer_watermark()
            watermarking.inserer()

        # Image originale en base64
        original_b64 = image_to_b64(watermarking.image_originale)

        # Image tatouée sauvegardée + base64
        tatouee_path = os.path.join(UPLOAD_FOLDER, 'tatouee.jpg')
        cv2.imwrite(tatouee_path, watermarking.image_tatouee.astype(np.uint8))
        tatouee_b64 = image_to_b64(watermarking.image_tatouee)

        # Métriques
        metriques = watermarking.evaluer(watermarking.image_tatouee)

        return jsonify({
            'original': f'data:image/jpeg;base64,{original_b64}',
            'tatouee': f'data:image/jpeg;base64,{tatouee_b64}',
            'psnr': round(metriques['PSNR'], 3),
            'ssim': round(metriques['SSIM'], 4),
            'ber': round(metriques['BER'], 4),
            'accuracy': round(metriques['Accuracy'], 4),
            'taille_wm': int(len(watermarking.watermark)),
            'dimensions': {
                'height': watermarking.image_originale.shape[0],
                'width': watermarking.image_originale.shape[1]
            }
        })

    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': f'Erreur de traitement : {str(e)}'}), 500


@app.route('/attack', methods=['POST'])
def attack():
    global watermarking

    if watermarking is None:
        return jsonify({'error': 'Aucune image chargée. Uploadez une image d\'abord.'}), 400

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Données JSON invalides'}), 400

    type_attack = data.get('type', '')

    try:
        with _lock:
            if type_attack == 'bruit':
                sigma = float(data.get('sigma', 10))
                sigma = max(0, min(100, sigma))  # Clamp sécurisé
                image_attaquee = watermarking.attaque_bruit(watermarking.image_tatouee, sigma)
                nom = f"Bruit gaussien (σ={sigma})"

            elif type_attack == 'jpeg':
                qualite = int(data.get('qualite', 50))
                qualite = max(1, min(100, qualite))
                image_attaquee = watermarking.attaque_jpeg(watermarking.image_tatouee, qualite)
                nom = f"Compression JPEG (Q={qualite})"

            elif type_attack == 'crop':
                pourcentage = float(data.get('pourcentage', 0.9))
                pourcentage = max(0.3, min(1.0, pourcentage))
                image_attaquee = watermarking.attaque_crop(watermarking.image_tatouee, pourcentage)
                nom = f"Recadrage ({pourcentage*100:.0f}%)"

            elif type_attack == 'rotation':
                angle = float(data.get('angle', 5))
                angle = max(-30, min(30, angle))
                image_attaquee = watermarking.attaque_rotation(watermarking.image_tatouee, angle)
                nom = f"Rotation ({angle}°)"

            elif type_attack == 'flou':
                ksize = int(data.get('ksize', 5))
                ksize = max(1, min(51, ksize))
                image_attaquee = watermarking.attaque_flou(watermarking.image_tatouee, ksize)
                nom = f"Flou gaussien ({ksize}×{ksize})"

            else:
                return jsonify({'error': f'Attaque inconnue : {type_attack}'}), 400

        # Sauvegarde
        path = os.path.join(UPLOAD_FOLDER, 'attaquee.jpg')
        cv2.imwrite(path, image_attaquee.astype(np.uint8))

        # Métriques après attaque
        metriques = watermarking.evaluer(image_attaquee)
        attaquee_b64 = image_to_b64(image_attaquee)

        return jsonify({
            'image': f'data:image/jpeg;base64,{attaquee_b64}',
            'ber': round(metriques['BER'], 4),
            'psnr': round(metriques['PSNR'], 3),
            'ssim': round(metriques['SSIM'], 4),
            'accuracy': round(metriques['Accuracy'], 4),
            'type': nom
        })

    except Exception as e:
        return jsonify({'error': f'Erreur lors de l\'attaque : {str(e)}'}), 500


@app.route('/reset', methods=['POST'])
def reset():
    """Réinitialise l'état de la session"""
    global watermarking
    with _lock:
        watermarking = None
    return jsonify({'message': 'État réinitialisé'})


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
