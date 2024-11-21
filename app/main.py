from fastapi import FastAPI
from app.routes import clustering_router

# Initialize FastAPI app
app = FastAPI()

# Include the clustering router
app.include_router(clustering_router)