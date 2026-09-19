from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import analyze

app = FastAPI(
    title="Medical Image Analysis",
    description="Service 2 of the medical AI system. Multi-label disease classification from medical images.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router)
