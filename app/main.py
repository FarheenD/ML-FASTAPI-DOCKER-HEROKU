from fastapi import FastAPI
from pydantic import BaseModel #Pydantic enforces type hints at runtime, and provides user-friendly errors when data is invalid
from app.model.model import predict_pipeline
from app.model.model import __version__ as model_version

app=FastAPI()


#Define your text to predict
class TextIn(BaseModel):
    text: str

#Define your output
class PredictionOut(BaseModel):
    language: str

@app.get('/')
def home():
    return {"health_check":"OK", "model_version":model_version}

@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    language = predict_pipeline(payload.text)
    return {"language": language}
