from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import predict

app = FastAPI(
    title="Disease Diagnosis Assistant",
    description="Service 1 of the medical AI system. Predicts a likely disease from patient symptoms.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # this service normally only receives traffic from the Go gateway
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
