import uuid
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db.models import Agent, Admin
from app.core.security import decode_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """Decode JWT and return {"id": UUID, "role": "agent"|"admin"}."""
    try:
        payload = decode_token(token)
        user_id = payload.get("sub")
        role = payload.get("role")
        if not user_id or role not in ("agent", "admin"):
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"id": uuid.UUID(user_id), "role": role}
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def require_agent(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Agent:
    if current_user["role"] != "agent":
        raise HTTPException(status_code=403, detail="Agent access required")
    agent = await db.get(Agent, current_user["id"])
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


async def require_admin(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Admin:
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    admin = await db.get(Admin, current_user["id"])
    if not admin:
        raise HTTPException(status_code=404, detail="Admin not found")
    return admin
