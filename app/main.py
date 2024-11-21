from fastapi import FastAPI
from app.routes import clustering_router
from fastapi.staticfiles import StaticFiles
import sys

sys.dont_write_bytecode = True

# Initialize FastAPI app
app = FastAPI()

# Include the clustering router
app.include_router(clustering_router)

app.mount("/static", StaticFiles(directory="static"), name="static")
