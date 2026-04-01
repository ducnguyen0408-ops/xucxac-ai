from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import numpy as np

app = FastAPI()

data = []
model = xgb.XGBClassifier()

class Input(BaseModel):
    sum:int
    even:int
    tai:int

@app.post("/predict")
def predict(i: Input):

    global data

    data.append([i.sum, i.even, i.tai])

    if len(data) > 20:
        X = np.array([d[:2] for d in data])
        y = np.array([d[2] for d in data])

        model.fit(X, y)

        prob = model.predict_proba([[i.sum, i.even]])[0][1]

        return {"tai": float(prob), "xiu": float(1-prob)}

    return {"msg":"learning"}
