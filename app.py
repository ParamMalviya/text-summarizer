import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from starlette.responses import RedirectResponse

from textSummarizer.logger import setup_logging, logger
from textSummarizer.exception import CustomException
from textSummarizer.pipelines.prediction import PredictionPipeline

setup_logging()   # wire up logging before the server starts

app = FastAPI()


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
        obj = PredictionPipeline()
        summary = obj.predict(text)
        return summary
    except Exception as e:
        logger.error(str(CustomException(e, sys)))
        return Response(f"Error Occurred! {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)