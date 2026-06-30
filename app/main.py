import os
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from predictor import predict_stress

# ==========================================
# FastAPI App
# ==========================================

app = FastAPI(
    title="XO Acoustica API",
    version="1.0",
    description="Machine Learning Speech Stress Analysis API"
)

# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Home
# ==========================================

@app.get("/")
def home():

    return {
        "Application": "XO Acoustica",
        "Version": "1.0",
        "Status": "Running"
    }


# ==========================================
# Health Check
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ==========================================
# Prediction Endpoint
# ==========================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    allowed = [".wav", ".mp3", ".m4a"]

    extension = os.path.splitext(file.filename)[1].lower()

    if extension not in allowed:

        raise HTTPException(
            status_code=400,
            detail="Only WAV, MP3 and M4A files are supported."
        )

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp:

            temp.write(await file.read())

            temp_path = temp.name

        result = predict_stress(temp_path)

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        if temp_path and os.path.exists(temp_path):

            os.remove(temp_path)