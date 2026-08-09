import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response, RedirectResponse

from textSummarizer.logger import setup_logging, logger
from textSummarizer.exception import CustomException
from textSummarizer.pipelines.prediction import PredictionPipeline

setup_logging()   # wire up logging before the server starts

app = FastAPI()

# load the model once, lazily on the first /predict, then reuse it.
# building PredictionPipeline loads the ~2.2GB model, so doing it per request
# would make every call as slow as a cold start. this way only the first hit pays it
_prediction_pipeline = None


def get_prediction_pipeline():
    global _prediction_pipeline
    if _prediction_pipeline is None:
        _prediction_pipeline = PredictionPipeline()
    return _prediction_pipeline


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def training():
    try:
        os.system(f"{sys.executable} main.py")
        return Response("Training successful !!")
    except Exception as e:
        logger.error(str(CustomException(e, sys)))
        return Response(f"Error Occurred! {e}")


@app.post("/predict")
async def predict_route(text: str):
    try:
        summary = get_prediction_pipeline().predict(text)
        return summary
    except Exception as e:
        logger.error(str(CustomException(e, sys)))
        return Response(f"Error Occurred! {e}")


if __name__ == "__main__":
    # port 8000 so a local `python app.py` matches what the Streamlit UI calls
    # (the container doesn't use this line -- start.sh runs uvicorn on 8000 too)
    uvicorn.run(app, host="0.0.0.0", port=8000)