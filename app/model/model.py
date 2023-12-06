import pickle
import re
from pathlib import Path
import sklearn

__version__ = '0.1.0'

#BASE_DIR = Path(__file__).resolve(strict=True).parent
BASE_DIR=Path.cwd()
#model_path=r'C:\Users\fdair\Documents\ML apps\ML-FASTAPI-DOCKER-HEROKU\app\model\trained_pipeline-0.1.0.pkl'
with open(f"{BASE_DIR}/app/model/trained_pipeline-{__version__}.pkl","rb") as f:
    model=pickle.load(f)

classes= [
    "Arabic",
    "Danish",
    "Dutch",
    "English",
    "French",
    "German",
    "Greek",
    "Hindi",
    "Italian",
    "Kannada",
    "Malayalam",
    "Portugeese",
    "Russian",
    "Spanish",
    "Sweedish",
    "Tamil",
    "Turkish",
]


#Helper Function
def predict_pipeline(text):
    text1=re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]',' ',text)
    text1 = re.sub(r"[[]]", " ", text1)
    text1=re.sub('  ',' ',text1)
    text1=text1.lower()
    pred=model.predict([text1])
    return classes[pred[0]]