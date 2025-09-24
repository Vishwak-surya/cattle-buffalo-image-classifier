import os
import torch
from flask import Flask, request, render_template, flash, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import torchvision.transforms as T
import io
import base64

from src.atc_classifier.models.model import ATCModel
from src.atc_classifier.config import load_config

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
CONFIG_PATH = 'configs/default.yaml'
MODEL_PATH = 'models/best.pt'

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for model (loaded once)
model = None
device = None
transform = None
config = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the model once at startup"""
    global model, device, transform, config
    
    try:
        config = load_config(CONFIG_PATH)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = ATCModel(
            backbone_name=config.model.backbone,
            pretrained=False,
            num_classes=config.data.num_classes if config.model.classification_head else 0,
            regression_traits=config.model.regression_traits,
        ).to(device)
        
        if os.path.exists(MODEL_PATH):
            state = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state)
            model.eval()
            print(f"Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"Warning: Model file {MODEL_PATH} not found. Please train the model first.")
            
        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

def predict_image(image):
    """Make prediction on uploaded image"""
    if model is None:
        return {"error": "Model not loaded. Please ensure the model is trained and available."}
    
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Apply transforms
        x = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            out = model(x)
        
        result = {}
        
        if "class_logits" in out:
            logits = out["class_logits"]
            probabilities = logits.softmax(dim=-1).squeeze(0)
            pred = probabilities.argmax().item()
            
            result["species"] = "Buffalo" if pred == 1 else "Cattle"
            result["confidence"] = float(probabilities.max().item())
            result["probabilities"] = {
                "cattle": float(probabilities[0].item()),
                "buffalo": float(probabilities[1].item())
            }
            
        if "traits" in out:
            traits = out["traits"].cpu().squeeze(0).tolist()
            result["traits"] = traits
            
        return result
        
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        try:
            # Read image directly from memory
            image = Image.open(io.BytesIO(file.read()))
            
            # Make prediction
            result = predict_image(image)
            
            # Convert image to base64 for display
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='JPEG')
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            
            return render_template('result.html', 
                                 result=result, 
                                 image_data=img_str,
                                 filename=secure_filename(file.filename))
        
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            return redirect(url_for('index'))
    
    else:
        flash('Invalid file type. Please upload PNG, JPG, JPEG, or GIF files.')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    try:
        image = Image.open(io.BytesIO(file.read()))
        result = predict_image(image)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Loading model...")
    load_model()
    print("Starting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=5000)