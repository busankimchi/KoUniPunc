import uvicorn
import uvloop
import asyncio
from fastapi import FastAPI

app = FastAPI(title="KoUniPunc server", version="0.0.1", description="KoUniPunc server")


asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


@app.post("/inference", status_code=200)
def inference():
    """
    Inference KoUniPunc
    """


@app.get("/", status_code=200)
def root():
    """
    root API
    """
    return {"status": "ok"}
