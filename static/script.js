 
let currentImageData = null;

document.getElementById('uploadBtn').addEventListener('click', async () => {
    const fileInput = document.getElementById('imageInput');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Veuillez sélectionner une image');
        return;
    }
    
    const formData = new FormData();
    formData.append('image', file);
    
    showLoading(true);
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            document.getElementById('originalImg').src = data.original;
            document.getElementById('tatoueeImg').src = data.tatouee;
            document.getElementById('metrics').innerHTML = `
                <strong>📊 Métriques :</strong><br>
                PSNR: ${data.psnr.toFixed(2)} dB<br>
                BER: ${data.ber.toFixed(4)}<br>
                Taille watermark: ${data.taille_wm} bits
            `;
            document.getElementById('imagesContainer').style.display = 'flex';
            document.getElementById('attackSection').style.display = 'block';
            currentImageData = data;
        } else {
            alert('Erreur: ' + data.error);
        }
    } catch (error) {
        alert('Erreur de connexion: ' + error);
    }
    
    showLoading(false);
});

async function applyAttack(type) {
    showLoading(true);
    document.getElementById('attackCard').style.display = 'block';
    
    let params = {};
    
    if (type === 'bruit') {
        params = { type, sigma: 10 };
    } else if (type === 'jpeg') {
        params = { type, qualite: 70 };
    } else if (type === 'crop') {
        params = { type, pourcentage: 0.9 };
    } else if (type === 'rotation') {
        params = { type, angle: 5 };
    }
    
    try {
        const response = await fetch('/attack', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });
        
        const data = await response.json();
        
        if (response.ok) {
            document.getElementById('attaqueeImg').src = data.image;
            document.getElementById('attackMetrics').innerHTML = `
                <strong>🎯 Attaque : ${data.type}</strong><br>
                BER après attaque: ${data.ber.toFixed(4)}
            `;
        } else {
            alert('Erreur: ' + data.error);
        }
    } catch (error) {
        alert('Erreur: ' + error);
    }
    
    showLoading(false);
}

function showLoading(show) {
    document.getElementById('loading').style.display = show ? 'block' : 'none';
}