import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Multi-Model Potato Disease Classification API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATHS = {
    "TR_SE_NET": "/mnt/c/development/Thesis/PotatoDiseaseClassification-CNN/models/plant_tr_se_net_v1.keras",
    "CNN_Baseline": "/mnt/c/development/Thesis/PotatoDiseaseClassification-CNN/models/model_v101.keras",
    "Parallel Depthwise(PD)CNN":"/mnt/c/development/Thesis/PotatoDiseaseClassification-CNN/models/Parallel_CNN_v1.keras" 
}

models = {}
for name, path in MODEL_PATHS.items():
    print(f"Loading {name}...")
    models[name] = tf.keras.models.load_model(path)

CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
IMAGE_SIZE = 256

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(io.BytesIO(data)).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.array(image)

@app.get("/ping")
async def ping():
    return "API is running with multiple models"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = read_file_as_image(image_bytes)
    img_batch = np.expand_dims(image, 0)
    all_results = {}
    for model_name, model in models.items():
        predictions = model.predict(img_batch)
        predicted_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        all_results[model_name] = {
            "predicted_class": CLASS_NAMES[predicted_index],
            "confidence": f"{round(confidence * 100, 2)}%"
        }
    return {
        "filename": file.filename,
        "predictions": all_results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)