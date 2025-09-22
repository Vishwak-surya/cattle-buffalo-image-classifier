from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
import os

# Placeholder import; wire into real inference when model/weights are ready
from ..infer import predict as run_predict

app = FastAPI(title="ATC Classifier API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # These paths assume defaults; replace with configured paths as needed
        # Returns JSON via stdout; in production, refactor to return dict directly
        # For now, respond with a simple OK while scaffolding is completed
        return JSONResponse({"message": "Prediction endpoint scaffolded", "temp_file": tmp_path})
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
