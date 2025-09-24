# Cattle vs Buffalo Classifier - Web Application

This web application provides an easy-to-use interface for classifying images of cattle and buffalo using your trained deep learning model.

## Features

- **Drag & Drop Interface**: Simple web interface for uploading images
- **Real-time Predictions**: Get instant classification results
- **Confidence Scores**: View probability breakdown for each class
- **Responsive Design**: Works on desktop and mobile devices
- **API Endpoint**: REST API for programmatic access

## Quick Start

### Option 1: Using PowerShell (Recommended for Windows)
```powershell
./start_web.ps1
```

### Option 2: Using Batch File
```cmd
start_web.bat
```

### Option 3: Manual Start
```cmd
# Activate virtual environment
.venv\Scripts\activate

# Install web dependencies (if not already installed)
pip install Flask Werkzeug

# Start the application
python app.py
```

## Accessing the Web Application

Once started, open your web browser and go to:
```
http://localhost:5000
```

## Using the Web Interface

1. **Upload Image**: Click "Choose Image File" or drag and drop an image
2. **Classify**: Click "Classify Image" to get predictions
3. **View Results**: See the classification result with confidence scores
4. **Upload Another**: Click "Upload Another Image" to classify more images

## Supported File Formats

- PNG (.png)
- JPEG (.jpg, .jpeg)
- GIF (.gif)
- Maximum file size: 16MB

## API Usage

You can also use the REST API endpoint for programmatic access:

```bash
curl -X POST -F "file=@your_image.jpg" http://localhost:5000/api/predict
```

Response format:
```json
{
  "species": "Cattle",
  "confidence": 0.95,
  "probabilities": {
    "cattle": 0.95,
    "buffalo": 0.05
  }
}
```

## Model Requirements

Make sure you have:
1. A trained model file at `models/best.pt`
2. Configuration file at `configs/default.yaml`
3. All required dependencies installed

## Troubleshooting

### Model Not Found
If you see "Model not loaded" error:
1. Train your model first using `python src/atc_classifier/train.py`
2. Ensure the model file exists at `models/best.pt`

### Dependencies Missing
If you get import errors:
1. Activate your virtual environment: `.venv\Scripts\activate`
2. Install requirements: `pip install -r requirements.txt`

### Port Already in Use
If port 5000 is busy:
1. Stop any running Flask applications
2. Or modify the port in `app.py` (change the last line)

## File Structure

```
├── app.py              # Main Flask application
├── templates/          # HTML templates
│   ├── base.html      # Base template
│   ├── index.html     # Upload form
│   └── result.html    # Results page
├── static/            # CSS and static files
│   └── style.css      # Custom styling
├── start_web.bat      # Windows batch startup script
├── start_web.ps1      # PowerShell startup script
└── WEB_README.md      # This file
```

## Development

To modify the web application:

1. **Frontend**: Edit HTML templates in `templates/` and CSS in `static/style.css`
2. **Backend**: Modify `app.py` for new routes or functionality
3. **Styling**: Update `static/style.css` for visual changes

The application uses:
- **Flask**: Web framework
- **Bootstrap 5**: CSS framework
- **Font Awesome**: Icons
- **Custom CSS**: Additional styling

## Security Notes

This application is configured for development/local use. For production deployment:

1. Change the secret key in `app.py`
2. Set `debug=False`
3. Use a production WSGI server (gunicorn, waitress, etc.)
4. Add proper error handling and validation
5. Implement file upload limits and security checks