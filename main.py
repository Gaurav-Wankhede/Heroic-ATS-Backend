from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import ats_router
import uvicorn
import os

app = FastAPI()

origins = [
    "http://localhost:3000",
    "https://heroic-ats-frontend.vercel.app",
    "https://heroic-ats-platform-738207385737.us-central1.run.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"]   # Allow all headers
)

app.include_router(ats_router.router)