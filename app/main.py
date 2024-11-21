from fastapi import FastAPI
from app.routes import clustering_router

# Initialize FastAPI app
app = FastAPI()

# Include the clustering router
app.include_router(clustering_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)