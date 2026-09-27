from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import ask

app = FastAPI(
    title="Medical Chatbot",
    description="Service 4 of the medical AI system. Educational Q&A only — not diagnosis or prescribing.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ask.router)