from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import ats_router
import uvicorn
import os

app = FastAPI()

origins = [
    "http://localhost:3000",
    "localhost:3000",
    "https://heroic-ats-frontend.vercel.app/",
    
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(ats_router.router)