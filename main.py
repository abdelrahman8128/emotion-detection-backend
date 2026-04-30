import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select

from app.db.database import engine, AsyncSessionLocal
from app.db.models import Base, Admin
from app.core.security import hash_password
from app.api import auth, calls, agents, admin, demo


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Create all tables on startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Seed default admin if none exists
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Admin).limit(1))
        if result.scalar_one_or_none() is None:
            username = os.getenv("ADMIN_USERNAME", "admin")
            password = os.getenv("ADMIN_PASSWORD", "admin")
            db.add(Admin(username=username, password_hash=hash_password(password)))
            await db.commit()
            print(f"[STARTUP] Default admin created  —  username: {username}  password: {password}")
            print("[STARTUP] Change these via ADMIN_USERNAME / ADMIN_PASSWORD env vars.")

    yield


app = FastAPI(
    title="Call Center Emotion Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Flutter Web and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(calls.router)
app.include_router(agents.router)   # /me/*
app.include_router(admin.router)    # /admin/*
app.include_router(demo.router)     # /demo/*


app.mount("/static", StaticFiles(directory="static"), name="static")



@app.get("/")
async def root():
    return {"status": "ok", "message": "Call Center Emotion Detection API"}
