from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import base64
import os
import threading
import json
import time
from datetime import datetime
from watermarking import WatermarkingDCT

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

_lock = threading.Lock()

# Sessions multiples pour comparaison
sessions = {}  # session_id -> WatermarkingDCT
history = []   # liste des tests effectués


def image_to_b64(image_array: np.ndarray) -> str:
    success, buffer = cv2.imencode('.jpg', image_array.astype(np.uint8),
                                   [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not success:
        raise ValueError("Échec de l'encodage")
    return base64.b64encode(buffer).decode()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'Aucune image'}), 400

    file = request.files['image']
    slot = request.form.get('slot', 'A')  # slot A ou B pour comparaison
    delta = int(request.form.get('delta', 30))
    seed = int(request.form.get('seed', 42))

    if not file or file.filename == '':
        return jsonify({'error': 'Fichier invalide'}), 400

    path = os.path.join(UPLOAD_FOLDER, f'upload_{slot}.jpg')
    try:
        file.save(path)
        with _lock:
            wm = WatermarkingDCT(path, delta=delta, seed=seed)
            wm.generer_watermark()
            wm.inserer()
            sessions[slot] = wm

        metriques = wm.evaluer(wm.image_tatouee)

        # Sauvegarder image tatouée
        tatouee_path = os.path.join(UPLOAD_FOLDER, f'tatouee_{slot}.jpg')
        cv2.imwrite(tatouee_path, wm.image_tatouee.astype(np.uint8))

        # Ajouter à l'historique
        history.append({
            'id': len(history) + 1,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'slot': slot,
            'filename': file.filename,
            'delta': delta,
            'psnr': round(metriques['PSNR'], 2),
            'ssim': round(metriques['SSIM'], 4),
            'ber': round(metriques['BER'], 4),
            'accuracy': round(metriques['Accuracy'] * 100, 1),
            'taille_wm': int(len(wm.watermark)),
            'attaque': 'Aucune',
            'dimensions': f"{wm.image_originale.shape[1]}×{wm.image_originale.shape[0]}"
        })

        return jsonify({
            'slot': slot,
            'original': f'data:image/jpeg;base64,{image_to_b64(wm.image_originale)}',
            'tatouee': f'data:image/jpeg;base64,{image_to_b64(wm.image_tatouee)}',
            'psnr': round(metriques['PSNR'], 3),
            'ssim': round(metriques['SSIM'], 4),
            'ber': round(metriques['BER'], 4),
            'accuracy': round(metriques['Accuracy'] * 100, 1),
            'taille_wm': int(len(wm.watermark)),
            'dimensions': f"{wm.image_originale.shape[1]}×{wm.image_originale.shape[0]}",
            'delta': delta,
            'nb_blocs': int(wm.nb_blocs)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/attack', methods=['POST'])
def attack():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Données invalides'}), 400

    slot = data.get('slot', 'A')
    wm = sessions.get(slot)
    if wm is None:
        return jsonify({'error': 'Aucune image chargée'}), 400

    type_attack = data.get('type', '')

    try:
        with _lock:
            if type_attack == 'bruit':
                sigma = max(0, min(100, float(data.get('sigma', 10))))
                image_attaquee = wm.attaque_bruit(wm.image_tatouee, sigma)
                nom = f"Bruit gaussien (σ={sigma})"
            elif type_attack == 'jpeg':
                qualite = max(1, min(100, int(data.get('qualite', 50))))
                image_attaquee = wm.attaque_jpeg(wm.image_tatouee, qualite)
                nom = f"JPEG (Q={qualite})"
            elif type_attack == 'crop':
                pct = max(0.3, min(1.0, float(data.get('pourcentage', 0.9))))
                image_attaquee = wm.attaque_crop(wm.image_tatouee, pct)
                nom = f"Recadrage ({pct*100:.0f}%)"
            elif type_attack == 'rotation':
                angle = max(-30, min(30, float(data.get('angle', 5))))
                image_attaquee = wm.attaque_rotation(wm.image_tatouee, angle)
                nom = f"Rotation ({angle}°)"
            elif type_attack == 'flou':
                ksize = int(data.get('ksize', 5))
                if ksize % 2 == 0: ksize += 1
                image_attaquee = wm.attaque_flou(wm.image_tatouee, ksize)
                nom = f"Flou ({ksize}×{ksize})"
            elif type_attack == 'contraste':
                alpha = max(0.5, min(3.0, float(data.get('alpha', 1.5))))
                image_attaquee = wm.attaque_contraste(wm.image_tatouee, alpha)
                nom = f"Contraste (×{alpha})"
            elif type_attack == 'median':
                ksize = int(data.get('ksize', 3))
                if ksize % 2 == 0: ksize += 1
                image_attaquee = wm.attaque_median(wm.image_tatouee, ksize)
                nom = f"Filtre médian ({ksize}×{ksize})"
            elif type_attack == 'sel_poivre':
                prob = max(0.01, min(0.3, float(data.get('prob', 0.05))))
                image_attaquee = wm.attaque_sel_poivre(wm.image_tatouee, prob)
                nom = f"Sel & Poivre ({prob*100:.0f}%)"
            else:
                return jsonify({'error': 'Attaque inconnue'}), 400

        path = os.path.join(UPLOAD_FOLDER, f'attaquee_{slot}.jpg')
        cv2.imwrite(path, image_attaquee.astype(np.uint8))
        metriques = wm.evaluer(image_attaquee)

        # Ajouter à l'historique
        history.append({
            'id': len(history) + 1,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'slot': slot,
            'filename': '—',
            'delta': wm.delta,
            'psnr': round(metriques['PSNR'], 2),
            'ssim': round(metriques['SSIM'], 4),
            'ber': round(metriques['BER'], 4),
            'accuracy': round(metriques['Accuracy'] * 100, 1),
            'taille_wm': int(len(wm.watermark)),
            'attaque': nom,
            'dimensions': f"{wm.image_originale.shape[1]}×{wm.image_originale.shape[0]}"
        })

        return jsonify({
            'image': f'data:image/jpeg;base64,{image_to_b64(image_attaquee)}',
            'ber': round(metriques['BER'], 4),
            'psnr': round(metriques['PSNR'], 3),
            'ssim': round(metriques['SSIM'], 4),
            'mse': round(float(np.mean((wm.image_originale - image_attaquee)**2)), 3),
            'accuracy': round(metriques['Accuracy'] * 100, 1),
            'type': nom,
            'slot': slot
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/scan_delta', methods=['POST'])
def scan_delta():
    """Scan de plusieurs valeurs de delta pour une image"""
    data = request.get_json(silent=True)
    slot = data.get('slot', 'A')
    wm = sessions.get(slot)
    if wm is None:
        return jsonify({'error': 'Aucune image chargée'}), 400

    deltas = [5, 10, 15, 20, 25, 30, 40, 50, 60, 80]
    results = []

    for delta in deltas:
        dct_img = wm.apply_dct_blocks(wm.image_originale)
        dct_mod = wm.inserer_qim(dct_img, wm.watermark, delta=delta)
        img_tat = wm.apply_idct_blocks(dct_mod)

        from skimage.metrics import peak_signal_noise_ratio as psnr_fn
        from skimage.metrics import structural_similarity as ssim_fn
        psnr_val = psnr_fn(wm.image_originale, img_tat, data_range=255)
        ssim_val = ssim_fn(wm.image_originale, img_tat, data_range=255)

        # Test robustesse avec JPEG Q=70
        img_att = wm.attaque_jpeg(img_tat, qualite=70)
        dct_att = wm.apply_dct_blocks(img_att)
        wm_ext = wm.extraire_qim(dct_att, wm.nb_blocs, delta=delta)
        ber = wm.calculer_ber(wm.watermark, wm_ext)

        results.append({
            'delta': delta,
            'psnr': round(psnr_val, 2),
            'ssim': round(ssim_val, 4),
            'ber': round(ber, 4),
            'accuracy': round((1 - ber) * 100, 1)
        })

    return jsonify({'results': results})


@app.route('/compare', methods=['POST'])
def compare():
    """Compare les métriques des slots A et B"""
    wm_a = sessions.get('A')
    wm_b = sessions.get('B')

    if wm_a is None or wm_b is None:
        return jsonify({'error': 'Chargez deux images (A et B) pour comparer'}), 400

    met_a = wm_a.evaluer(wm_a.image_tatouee)
    met_b = wm_b.evaluer(wm_b.image_tatouee)

    return jsonify({
        'A': {
            'psnr': round(met_a['PSNR'], 2),
            'ssim': round(met_a['SSIM'], 4),
            'ber': round(met_a['BER'], 4),
            'accuracy': round(met_a['Accuracy'] * 100, 1),
            'taille': int(len(wm_a.watermark)),
            'delta': wm_a.delta
        },
        'B': {
            'psnr': round(met_b['PSNR'], 2),
            'ssim': round(met_b['SSIM'], 4),
            'ber': round(met_b['BER'], 4),
            'accuracy': round(met_b['Accuracy'] * 100, 1),
            'taille': int(len(wm_b.watermark)),
            'delta': wm_b.delta
        }
    })


@app.route('/download/<slot>')
def download(slot):
    """Téléchargement de l'image tatouée"""
    path = os.path.join(UPLOAD_FOLDER, f'tatouee_{slot}.jpg')
    if not os.path.exists(path):
        return jsonify({'error': 'Image non disponible'}), 404
    return send_file(path, as_attachment=True, download_name=f'watermarked_{slot}.jpg')


@app.route('/download_attacked/<slot>')
def download_attacked(slot):
    """Téléchargement de l'image attaquée"""
    path = os.path.join(UPLOAD_FOLDER, f'attaquee_{slot}.jpg')
    if not os.path.exists(path):
        return jsonify({'error': 'Image non disponible'}), 404
    return send_file(path, as_attachment=True, download_name=f'attacked_{slot}.jpg')


@app.route('/history')
def get_history():
    return jsonify({'history': history[-20:]})  # 20 derniers


@app.route('/reset', methods=['POST'])
def reset():
    global history
    sessions.clear()
    history = []
    return jsonify({'message': 'Réinitialisé'})


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
