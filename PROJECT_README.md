# Potato Disease Classification - FastAPI Service

A simple web application for potato disease classification using two deep learning models: CNN and TR-SE-NET.

## Project Structure

```
PotatoDiseaseClassification-CNN/
├── frontend/                 # Web interface (moved to root)
│   ├── index.html
│   └── open_frontend.bat
├── fastapi/
│   ├── backend/
│   │   ├── main.py           # FastAPI application
│   │   └── run_server.py     # Server launcher
│   │   └── run_server.bat    # Windows server launcher
│   ├── README.md             # Full documentation
│   ├── SETUP.md              # Setup instructions
│   └── QUICKSTART.bat        # Quick start script
├── requirements.txt          # All dependencies (consolidated)
├── models/
│   ├── model_v101.keras      # CNN model
│   └── plant_tr_se_net_v1.keras  # TR-SE-NET model
└── ... (other files)
```

## Features

- **Dual Model Predictions**: Uses both CNN and TR-SE-NET models for disease classification
- **Image Upload**: Simple drag-and-drop or click-to-upload interface
- **Real-time Predictions**: Get predictions from both models instantly
- **Detailed Results**: View confidence scores and detailed probability distribution
- **Responsive Design**: Works on desktop and mobile devices
- **Simple UI**: Clean and straightforward interface

## Models

The application uses two pre-trained models located in the `models/` directory:

1. **CNN Model** (`model_v101.keras`) - Traditional Convolutional Neural Network
2. **TR-SE-NET Model** (`plant_tr_se_net_v1.keras`) - Transformer + Squeeze-and-Excitation Network

## Disease Classes

The models can classify the following potato diseases:

- Potato___Early_blight
- Potato___healthy
- Potato___Late_blight

## Prerequisites

- Python 3.8+
- pip (Python package manager)

## Installation & Setup

### 1. Install Dependencies (Root Level)

```bash
pip install -r requirements.txt
```

Or with your virtual environment:

```bash
source linux-venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Backend Server

```bash
cd fastapi/backend
python run_server.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload
```

Or on Windows, double-click:
```
fastapi\backend\run_server.bat
```

The API will be available at: `http://localhost:8000`

API documentation will be available at: `http://localhost:8000/docs`

### 3. Open the Frontend

Open the `frontend/index.html` file in a web browser. You can:

- Double-click the file
- Or double-click `frontend/open_frontend.bat`
- Or use a local server:

```bash
# Python 3+
python -m http.server 8001 --directory frontend
```

Then visit: `http://localhost:8001`

## API Endpoints

### POST `/predict-disease`

Upload an image and get predictions from both models.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Image file

**Response:**
```json
{
  "status": "success",
  "cnn": {
    "class_name": "Potato___healthy",
    "confidence": 95.67,
    "predictions": {
      "Potato___Early_blight": 2.33,
      "Potato___healthy": 95.67,
      "Potato___Late_blight": 2.00
    }
  },
  "trsenet": {
    "class_name": "Potato___healthy",
    "confidence": 98.45,
    "predictions": {
      "Potato___Early_blight": 0.50,
      "Potato___healthy": 98.45,
      "Potato___Late_blight": 1.05
    }
  }
}
```

### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

### GET `/`

Get API information and model status.

**Response:**
```json
{
  "message": "Potato Disease Classification API",
  "status": "running",
  "models": {
    "cnn": "loaded",
    "trsenet": "loaded"
  }
}
```

## Usage

1. Start the backend server
2. Open the frontend in your browser
3. Upload an image of a potato leaf
4. View the predictions from both models in the results table
5. Check detailed probability distributions for each model

## Technical Details

### Image Processing

- **Input Size**: 256x256 pixels
- **Color Space**: RGB (3 channels)
- **Normalization**: Pixel values scaled to 0-1 range

### CORS Support

The API includes CORS middleware to allow requests from the frontend running on different ports/domains.

## Troubleshooting

### Models not loading
- Ensure both `.keras` model files exist in the `models/` directory
- Check that model file paths are correct
- Verify TensorFlow is properly installed

### CORS errors
- The API is configured to allow all origins. If issues persist, check CORS configuration in `fastapi/backend/main.py`

### Image upload fails
- Ensure the image format is supported (JPEG, PNG, etc.)
- Check that the backend API is running and accessible
- Verify network connectivity between frontend and backend

## File Formats Supported

- JPEG (.jpg, .jpeg)
- PNG (.png)
- WebP (.webp)
- BMP (.bmp)
- GIF (.gif)

## Notes

- The first prediction may take a few seconds as the models are loaded
- Maximum recommended image size: 10 MB
- Both models run inference on the same image for comparison
- Confidence scores represent the probability distribution across all disease classes

## Dependencies

All dependencies are listed in the root `requirements.txt`:

- FastAPI==0.104.1
- Uvicorn==0.24.0
- TensorFlow>=2.13.0
- Pillow==10.0.1
- NumPy>=1.24.0
- python-multipart==0.0.6

## Future Enhancements

- Model ensemble voting
- Batch image processing
- Model performance metrics
- User feedback and model retraining
- Docker containerization
- Database for prediction history
