from fastapi import FastAPI
from app.routes import clustering_router
from fastapi.staticfiles import StaticFiles
import logging

logging.basicConfig(level=logging.DEBUG)

# Initialize FastAPI app
app = FastAPI()

# Include the clustering router
app.include_router(clustering_router)

app.mount("/static", StaticFiles(directory="static"), name="static")
