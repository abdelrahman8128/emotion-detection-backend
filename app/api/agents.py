"""Agent self-view endpoints — /me/*"""
import uuid
from datetime import datetime, date
from collections import defaultdict

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from app.db.database import get_db
from app.db.models import Agent, Call, EmotionSegment, Alert
from app.schemas.schemas import (
    MyProfile, MyPerformance, AgentStatsOut, EmotionBreakdown,
    CallsListResponse, CallSummary,
)
from app.core.deps import require_agent
from app.services.alerts import dominant_emotion, alert_level

router = APIRouter(prefix="/me", tags=["me"])

POSITIVE_EMOTIONS = {"happy", "neutral", "surprise"}


def _date_conditions(start_date: date | None, end_date: date | None) -> list:
    conds = []
    if start_date:
        conds.append(Call.start_time >= datetime(start_date.year, start_date.month, start_date.day))
    if end_date:
        conds.append(Call.start_time <= datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59))
    return conds


def _pages(total: int, page_size: int) -> int:
    return ((total - 1) // page_size + 1) if total else 0


def _emotion_breakdown(segments) -> list[EmotionBreakdown]:
    counts: dict[str, int] = defaultdict(int)
    for seg in segments:
        counts[seg.emotion] += 1
    total = sum(counts.values())
    return [
        EmotionBreakdown(
            emotion=e, count=c,
            percentage=round(c / total * 100, 2) if total else 0.0,
        )
        for e, c in sorted(counts.items(), key=lambda x: -x[1])
    ]


# ── GET /me/profile ──────────────────────────────────────────────────────────

@router.get("/profile", response_model=MyProfile)
async def get_my_profile(agent: Agent = Depends(require_agent)):
    return MyProfile(
        agent_id=agent.id, name=agent.name,
        username=agent.username, team=agent.team,
        created_at=agent.created_at,
    )


# ── GET /me/calls ────────────────────────────────────────────────────────────

@router.get("/calls", response_model=CallsListResponse)
async def get_my_calls(
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    status: str | None = Query(None, pattern="^(processing|done|error)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    agent: Agent = Depends(require_agent),
    db: AsyncSession = Depends(get_db),
):
    conditions = [Call.agent_id == agent.id]
    conditions += _date_conditions(start_date, end_date)
    if status:
        conditions.append(Call.status == status)

    stmt = select(Call).where(and_(*conditions))

    total = (await db.execute(select(func.count()).select_from(stmt.subquery()))).scalar()

    stmt = stmt.order_by(Call.start_time.desc()).offset((page - 1) * page_size).limit(page_size)
    calls = (await db.execute(stmt)).scalars().all()

    summaries = []
    for call in calls:
        segs = (await db.execute(
            select(EmotionSegment).where(
                and_(EmotionSegment.call_id == call.id, EmotionSegment.speaker == "customer")
            )
        )).scalars().all()
        anger_pct = sum(1 for s in segs if s.emotion == "angry") / len(segs) if segs else 0.0
        unresolved = (await db.execute(
            select(func.count(Alert.id)).where(and_(Alert.call_id == call.id, Alert.resolved == False))
        )).scalar()
        summaries.append(CallSummary(
            call_id=call.id, agent_name=agent.name,
            duration=call.duration,
            dominant_emotion=dominant_emotion(segs),
            alert_level=alert_level(bool(unresolved), anger_pct),
            status=call.status, start_time=call.start_time,
        ))

    return CallsListResponse(
        calls=summaries, total=total, page=page,
        page_size=page_size, total_pages=_pages(total, page_size),
    )


# ── GET /me/performance ──────────────────────────────────────────────────────

@router.get("/performance", response_model=MyPerformance)
async def get_my_performance(
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    agent: Agent = Depends(require_agent),
    db: AsyncSession = Depends(get_db),
):
    conditions = [Call.agent_id == agent.id]
    conditions += _date_conditions(start_date, end_date)

    call_ids = [
        c.id for c in
        (await db.execute(select(Call).where(and_(*conditions)))).scalars().all()
    ]

    customer_segs, agent_segs = [], []
    if call_ids:
        segs = (await db.execute(
            select(EmotionSegment).where(EmotionSegment.call_id.in_(call_ids))
        )).scalars().all()
        customer_segs = [s for s in segs if s.speaker == "customer"]
        agent_segs = [s for s in segs if s.speaker == "agent"]

    n = len(customer_segs)
    positive_rate = sum(1 for s in customer_segs if s.emotion in POSITIVE_EMOTIONS) / n if n else 0.0
    anger_rate = sum(1 for s in customer_segs if s.emotion == "angry") / n if n else 0.0

    escalations = sustained = 0
    if call_ids:
        alerts = (await db.execute(
            select(Alert).where(Alert.call_id.in_(call_ids))
        )).scalars().all()
        for a in alerts:
            if a.type == "escalation":
                escalations += 1
            elif a.type == "anger_sustained":
                sustained += 1

    return MyPerformance(
        stats=AgentStatsOut(
            total_calls=len(call_ids),
            positive_rate=round(positive_rate, 4),
            anger_rate=round(anger_rate, 4),
            escalations=escalations,
            sustained_anger_alerts=sustained,
        ),
        customer_emotion_breakdown=_emotion_breakdown(customer_segs),
        agent_emotion_breakdown=_emotion_breakdown(agent_segs),
    )
