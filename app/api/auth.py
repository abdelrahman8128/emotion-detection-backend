from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.database import get_db
from app.db.models import Admin, Agent
from app.core.security import verify_password, create_access_token
from app.schemas.schemas import LoginRequest, TokenResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    # Try admin
    result = await db.execute(select(Admin).where(Admin.username == body.username))
    admin = result.scalar_one_or_none()
    if admin and verify_password(body.password, admin.password_hash):
        token = create_access_token({"sub": str(admin.id), "role": "admin"})
        return TokenResponse(access_token=token, role="admin", user_id=admin.id)

    # Try agent
    result = await db.execute(select(Agent).where(Agent.username == body.username))
    agent = result.scalar_one_or_none()
    if agent and verify_password(body.password, agent.password_hash):
        token = create_access_token({"sub": str(agent.id), "role": "agent"})
        return TokenResponse(access_token=token, role="agent", user_id=agent.id)

    raise HTTPException(status_code=401, detail="Invalid username or password")
