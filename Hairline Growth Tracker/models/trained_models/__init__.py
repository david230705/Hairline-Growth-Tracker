# Check what models are available
from models.trained_models import list_available_models

available_models = list_available_models()
print(f"Available models: {available_models}")

# Check if a specific model exists
from models.trained_models import is_model_available

if is_model_available("face_detector.h5"):
    print("Face detector model is available!")