from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import recommend

app = FastAPI(
    title="Drug Recommendation Assistant",
    description="Service 3 of the medical AI system. Clinical decision support — not a prescribing tool.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommend.router)
