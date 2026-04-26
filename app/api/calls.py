import os
import uuid
import shutil
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.database import get_db, AsyncSessionLocal
from app.db.models import Call, Agent, EmotionSegment, Alert
from app.schemas.schemas import UploadResponse, TimelineResponse, EmotionPoint
from app.services.audio import split_and_segment
from app.services.prediction import predict_emotion
from app.services.alerts import check_alerts
from app.core.deps import get_current_user, require_agent

router = APIRouter(prefix="/calls", tags=["calls"])

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Background processing task ────────────────────────────────────────────────

async def _process_call(call_id: uuid.UUID, audio_path: str) -> None:
    async with AsyncSessionLocal() as db:
        try:
            agent_chunks, customer_chunks = split_and_segment(audio_path)

            total_chunks = max(len(agent_chunks), len(customer_chunks))
            duration = total_chunks * 3

            call = await db.get(Call, call_id)
            if call is None:
                return
            call.duration = duration
            call.end_time = datetime.utcnow()

            # Agent segments
            for i, chunk in enumerate(agent_chunks):
                emotion, confidence = predict_emotion(chunk)
                db.add(EmotionSegment(
                    call_id=call_id, speaker="agent",
                    second_start=i * 3, emotion=emotion, confidence=confidence,
                ))

            # Customer segments
            customer_timeline = []
            for i, chunk in enumerate(customer_chunks):
                emotion, confidence = predict_emotion(chunk)
                db.add(EmotionSegment(
                    call_id=call_id, speaker="customer",
                    second_start=i * 3, emotion=emotion, confidence=confidence,
                ))
                customer_timeline.append({"emotion": emotion, "confidence": confidence})

            await db.flush()

            # Alerts
            for alert_data in check_alerts(str(call_id), customer_timeline):
                db.add(Alert(
                    call_id=call_id, type=alert_data["type"],
                    description=alert_data["description"],
                    triggered_at=datetime.utcnow(),
                ))

            call.status = "done"
            await db.commit()

        except Exception as exc:
            await db.rollback()
            async with AsyncSessionLocal() as db2:
                call = await db2.get(Call, call_id)
                if call:
                    call.status = "error"
                    await db2.commit()
            raise exc

        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)


# ── POST /calls/upload ────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse, status_code=202)
async def upload_call(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    agent: Agent = Depends(require_agent),
    db: AsyncSession = Depends(get_db),
):
    # Save audio file
    filename = f"{uuid.uuid4()}_{audio_file.filename}"
    audio_path = os.path.join(UPLOAD_DIR, filename)
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio_file.file, f)

    # Create call record
    call = Call(agent_id=agent.id, status="processing")
    db.add(call)
    await db.commit()
    await db.refresh(call)

    background_tasks.add_task(_process_call, call.id, audio_path)
    return UploadResponse(call_id=call.id, status="processing")


# ── GET /calls/{call_id}/timeline ─────────────────────────────────────────────

@router.get("/{call_id}/timeline", response_model=TimelineResponse)
async def get_timeline(
    call_id: uuid.UUID,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    call = await db.get(Call, call_id)
    if call is None:
        raise HTTPException(status_code=404, detail="Call not found")

    # Agents can only view their own calls
    if current_user["role"] == "agent" and call.agent_id != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    result = await db.execute(
        select(EmotionSegment)
        .where(EmotionSegment.call_id == call_id)
        .order_by(EmotionSegment.second_start)
    )
    segments = result.scalars().all()

    agent_timeline = [
        EmotionPoint(second=s.second_start, emotion=s.emotion, confidence=s.confidence)
        for s in segments if s.speaker == "agent"
    ]
    customer_timeline = [
        EmotionPoint(second=s.second_start, emotion=s.emotion, confidence=s.confidence)
        for s in segments if s.speaker == "customer"
    ]

    return TimelineResponse(
        call_id=call_id, duration=call.duration,
        agent_timeline=agent_timeline, customer_timeline=customer_timeline,
    )
