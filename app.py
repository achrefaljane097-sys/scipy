from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import os
from watermarking import WatermarkingDCT

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Variable globale pour l'image courante
current_image = None
watermarking = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global watermarking, current_image
    
    if 'image' not in request.files:
        return jsonify({'error': 'Aucune image'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Nom vide'}), 400
    
    # Sauvegarde de l'image
    path = os.path.join(UPLOAD_FOLDER, 'upload.jpg')
    file.save(path)
    
    # Initialisation du watermarking
    watermarking = WatermarkingDCT(path)
    watermarking.generer_watermark()
    watermarking.inserer()
    
    # Encodage base64 pour affichage
    with open(path, 'rb') as f:
        original_b64 = base64.b64encode(f.read()).decode()
    
    # Image tatouée
    tatouee_path = os.path.join(UPLOAD_FOLDER, 'tatouee.jpg')
    cv2.imwrite(tatouee_path, watermarking.image_tatouee)
    with open(tatouee_path, 'rb') as f:
        tatouee_b64 = base64.b64encode(f.read()).decode()
    
    # Métriques
    wm_extrait = watermarking.extraire(watermarking.image_tatouee)
    ber = watermarking.calculer_ber(watermarking.watermark, wm_extrait)
    psnr_val = cv2.PSNR(watermarking.image_originale, watermarking.image_tatouee)
    
    return jsonify({
        'original': f'data:image/jpeg;base64,{original_b64}',
        'tatouee': f'data:image/jpeg;base64,{tatouee_b64}',
        'psnr': psnr_val,
        'ber': ber,
        'taille_wm': len(watermarking.watermark)
    })

@app.route('/attack', methods=['POST'])
def attack():
    global watermarking
    
    data = request.json
    type_attack = data.get('type', 'bruit')
    
    if type_attack == 'bruit':
        sigma = data.get('sigma', 10)
        image_attaquee = watermarking.attaque_bruit(watermarking.image_tatouee, sigma)
        nom = f"Bruit (σ={sigma})"
    
    elif type_attack == 'jpeg':
        qualite = data.get('qualite', 50)
        image_attaquee = watermarking.attaque_jpeg(watermarking.image_tatouee, qualite)
        nom = f"JPEG (Q={qualite})"
    
    elif type_attack == 'crop':
        pourcentage = data.get('pourcentage', 0.9)
        image_attaquee = watermarking.attaque_crop(watermarking.image_tatouee, pourcentage)
        nom = f"Crop ({pourcentage*100}%)"
    
    elif type_attack == 'rotation':
        angle = data.get('angle', 5)
        image_attaquee = watermarking.attaque_rotation(watermarking.image_tatouee, angle)
        nom = f"Rotation ({angle}°)"
    
    else:
        return jsonify({'error': 'Attaque inconnue'}), 400
    
    # Sauvegarde
    path = os.path.join(UPLOAD_FOLDER, 'attaquee.jpg')
    cv2.imwrite(path, image_attaquee)
    
    # Extraction
    wm_extrait = watermarking.extraire(image_attaquee)
    ber = watermarking.calculer_ber(watermarking.watermark, wm_extrait)
    
    # Encodage
    with open(path, 'rb') as f:
        attaquee_b64 = base64.b64encode(f.read()).decode()
    
    return jsonify({
        'image': f'data:image/jpeg;base64,{attaquee_b64}',
        'ber': ber,
        'type': nom
    })

if __name__ == '__main__':
    app.run(debug=True) 
