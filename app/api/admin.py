import uuid
from datetime import datetime, date
from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, cast
from sqlalchemy.types import Date as SADate

from app.db.database import get_db
from app.db.models import Admin, Agent, Call, EmotionSegment, Alert
from app.schemas.schemas import (
    CreateAgentRequest, AgentCreatedResponse,
    AgentListItem, AgentsListResponse,
    AgentFullProfile, AgentStatsOut, EmotionBreakdown,
    AdminDashboardStats, CallsPerDay, AgentRanking,
    CallSummary, CallsListResponse, AlertSummary,
)
from app.core.deps import require_admin
from app.core.security import hash_password
from app.services.alerts import dominant_emotion, alert_level

router = APIRouter(prefix="/admin", tags=["admin"])

POSITIVE_EMOTIONS = {"happy", "neutral", "surprise"}


# ── helpers ───────────────────────────────────────────────────────────────────

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


# ── POST /admin/agents ───────────────────────────────────────────────────────

@router.post("/agents", response_model=AgentCreatedResponse, status_code=201)
async def create_agent(
    body: CreateAgentRequest,
    db: AsyncSession = Depends(get_db),
    _: Admin = Depends(require_admin),
):
    existing = await db.execute(select(Agent).where(Agent.username == body.username))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Username already taken")

    agent = Agent(
        username=body.username,
        password_hash=hash_password(body.password),
        name=body.name,
        team=body.team,
    )
    db.add(agent)
    await db.commit()
    await db.refresh(agent)
    return AgentCreatedResponse(
        agent_id=agent.id, username=agent.username,
        name=agent.name, team=agent.team, created_at=agent.created_at,
    )


# ── GET /admin/agents ────────────────────────────────────────────────────────

@router.get("/agents", response_model=AgentsListResponse)
async def list_agents(
    search: str | None = Query(None, description="Filter by name or team"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    _: Admin = Depends(require_admin),
):
    stmt = select(Agent)
    if search:
        pattern = f"%{search}%"
        stmt = stmt.where(Agent.name.ilike(pattern) | Agent.team.ilike(pattern))

    total = (await db.execute(select(func.count()).select_from(stmt.subquery()))).scalar()

    stmt = stmt.order_by(Agent.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
    agents = (await db.execute(stmt)).scalars().all()

    items = []
    for agent in agents:
        n = (await db.execute(
            select(func.count(Call.id)).where(Call.agent_id == agent.id)
        )).scalar() or 0
        items.append(AgentListItem(
            agent_id=agent.id, name=agent.name, username=agent.username,
            team=agent.team, total_calls=n, created_at=agent.created_at,
        ))

    return AgentsListResponse(
        agents=items, total=total, page=page,
        page_size=page_size, total_pages=_pages(total, page_size),
    )


# ── GET /admin/agents/{id}/profile ───────────────────────────────────────────

@router.get("/agents/{agent_id}/profile", response_model=AgentFullProfile)
async def get_agent_profile(
    agent_id: uuid.UUID,
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    _: Admin = Depends(require_admin),
):
    agent = await db.get(Agent, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    date_conds = _date_conditions(start_date, end_date)

    # All calls for agent in period
    call_stmt = select(Call).where(Call.agent_id == agent_id)
    if date_conds:
        call_stmt = call_stmt.where(and_(*date_conds))
    all_calls = (await db.execute(call_stmt.order_by(Call.start_time.desc()))).scalars().all()
    call_ids = [c.id for c in all_calls]

    # Emotion segments in one query
    customer_segs, agent_segs = [], []
    all_segs = []
    if call_ids:
        seg_result = await db.execute(
            select(EmotionSegment).where(EmotionSegment.call_id.in_(call_ids))
        )
        all_segs = seg_result.scalars().all()
        customer_segs = [s for s in all_segs if s.speaker == "customer"]
        agent_segs = [s for s in all_segs if s.speaker == "agent"]

    # Alerts
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

    # Stats
    n_cust = len(customer_segs)
    positive_rate = sum(1 for s in customer_segs if s.emotion in POSITIVE_EMOTIONS) / n_cust if n_cust else 0.0
    anger_rate = sum(1 for s in customer_segs if s.emotion == "angry") / n_cust if n_cust else 0.0

    stats = AgentStatsOut(
        total_calls=len(call_ids),
        positive_rate=round(positive_rate, 4),
        anger_rate=round(anger_rate, 4),
        escalations=escalations,
        sustained_anger_alerts=sustained,
    )

    # Paginated calls
    total_count = len(all_calls)
    page_calls = all_calls[(page - 1) * page_size : page * page_size]

    call_summaries = []
    for call in page_calls:
        c_segs = [s for s in all_segs if s.call_id == call.id and s.speaker == "customer"]
        c_anger = sum(1 for s in c_segs if s.emotion == "angry") / len(c_segs) if c_segs else 0.0
        unresolved = (await db.execute(
            select(func.count(Alert.id)).where(and_(Alert.call_id == call.id, Alert.resolved == False))
        )).scalar()
        call_summaries.append(CallSummary(
            call_id=call.id, agent_name=agent.name,
            duration=call.duration,
            dominant_emotion=dominant_emotion(c_segs),
            alert_level=alert_level(bool(unresolved), c_anger),
            status=call.status, start_time=call.start_time,
        ))

    return AgentFullProfile(
        agent_id=agent.id, name=agent.name, username=agent.username,
        team=agent.team, created_at=agent.created_at,
        stats=stats,
        customer_emotion_breakdown=_emotion_breakdown(customer_segs),
        agent_emotion_breakdown=_emotion_breakdown(agent_segs),
        calls=call_summaries,
        total_calls_count=total_count, page=page,
        page_size=page_size, total_pages=_pages(total_count, page_size),
    )


# ── GET /admin/dashboard ─────────────────────────────────────────────────────

@router.get("/dashboard", response_model=AdminDashboardStats)
async def get_dashboard(
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    db: AsyncSession = Depends(get_db),
    _: Admin = Depends(require_admin),
):
    date_conds = _date_conditions(start_date, end_date)

    # Calls in period
    call_stmt = select(Call)
    if date_conds:
        call_stmt = call_stmt.where(and_(*date_conds))
    calls = (await db.execute(call_stmt)).scalars().all()
    call_ids = [c.id for c in calls]
    total_calls = len(calls)
    active_agents = len({c.agent_id for c in calls})

    # Calls per day
    cpd_stmt = (
        select(cast(Call.start_time, SADate).label("day"), func.count(Call.id).label("cnt"))
        .group_by(cast(Call.start_time, SADate))
        .order_by(cast(Call.start_time, SADate))
    )
    if date_conds:
        cpd_stmt = cpd_stmt.where(and_(*date_conds))
    calls_per_day = [
        CallsPerDay(date=str(r.day), count=r.cnt)
        for r in (await db.execute(cpd_stmt)).all()
    ]

    # Emotion distribution (customer)
    emotion_distribution: list[EmotionBreakdown] = []
    if call_ids:
        ed = await db.execute(
            select(EmotionSegment.emotion, func.count(EmotionSegment.id).label("cnt"))
            .where(and_(EmotionSegment.call_id.in_(call_ids), EmotionSegment.speaker == "customer"))
            .group_by(EmotionSegment.emotion)
        )
        rows = ed.all()
        total_segs = sum(r.cnt for r in rows)
        emotion_distribution = [
            EmotionBreakdown(
                emotion=r.emotion, count=r.cnt,
                percentage=round(r.cnt / total_segs * 100, 2) if total_segs else 0.0,
            )
            for r in sorted(rows, key=lambda x: -x.cnt)
        ]

    # Per-call anger + satisfaction
    angry_calls = 0
    total_anger = 0.0
    total_positive = 0.0
    if call_ids:
        pc = await db.execute(
            select(EmotionSegment.call_id, EmotionSegment.emotion, func.count(EmotionSegment.id).label("cnt"))
            .where(and_(EmotionSegment.call_id.in_(call_ids), EmotionSegment.speaker == "customer"))
            .group_by(EmotionSegment.call_id, EmotionSegment.emotion)
        )
        call_emo: dict[uuid.UUID, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for row in pc.all():
            call_emo[row.call_id][row.emotion] = row.cnt

        for ecounts in call_emo.values():
            t = sum(ecounts.values())
            if t:
                ap = ecounts.get("angry", 0) / t
                pp = sum(ecounts.get(e, 0) for e in POSITIVE_EMOTIONS) / t
                total_anger += ap
                total_positive += pp
                if ap > 0.30:
                    angry_calls += 1

    avg_anger = total_anger / total_calls if total_calls else 0.0
    avg_satisfaction = total_positive / total_calls if total_calls else 0.0

    # Alerts
    alert_summaries: list[AlertSummary] = []
    unresolved_count = 0
    if call_ids:
        al = await db.execute(
            select(Alert)
            .where(and_(Alert.call_id.in_(call_ids), Alert.resolved == False))
            .order_by(Alert.triggered_at.desc())
        )
        unresolved = al.scalars().all()
        unresolved_count = len(unresolved)
        alert_summaries = [
            AlertSummary(call_id=a.call_id, type=a.type, description=a.description, triggered_at=a.triggered_at)
            for a in unresolved[:10]
        ]

    # Top agents (by satisfaction) and worst (by anger)
    top_satisfaction: list[AgentRanking] = []
    top_anger: list[AgentRanking] = []
    if call_ids:
        ap_result = await db.execute(
            select(Call.agent_id, EmotionSegment.emotion, func.count(EmotionSegment.id).label("cnt"))
            .join(EmotionSegment, EmotionSegment.call_id == Call.id)
            .where(and_(Call.id.in_(call_ids), EmotionSegment.speaker == "customer"))
            .group_by(Call.agent_id, EmotionSegment.emotion)
        )
        agent_emo: dict[uuid.UUID, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for row in ap_result.all():
            agent_emo[row.agent_id][row.emotion] = row.cnt

        agent_call_counts: dict[uuid.UUID, int] = defaultdict(int)
        for c in calls:
            agent_call_counts[c.agent_id] += 1

        sat_list, ang_list = [], []
        for aid, ecounts in agent_emo.items():
            t = sum(ecounts.values())
            if not t:
                continue
            agent = await db.get(Agent, aid)
            name = agent.name if agent else "Unknown"
            nc = agent_call_counts[aid]
            sat_list.append(AgentRanking(agent_id=aid, agent_name=name,
                                         value=round(sum(ecounts.get(e, 0) for e in POSITIVE_EMOTIONS) / t, 4),
                                         total_calls=nc))
            ang_list.append(AgentRanking(agent_id=aid, agent_name=name,
                                         value=round(ecounts.get("angry", 0) / t, 4),
                                         total_calls=nc))
        top_satisfaction = sorted(sat_list, key=lambda x: -x.value)[:5]
        top_anger = sorted(ang_list, key=lambda x: -x.value)[:5]

    return AdminDashboardStats(
        period_start=str(start_date) if start_date else "all",
        period_end=str(end_date) if end_date else "all",
        total_calls=total_calls,
        active_agents=active_agents,
        angry_calls=angry_calls,
        avg_anger_pct=round(avg_anger, 4),
        avg_satisfaction_score=round(avg_satisfaction, 4),
        total_unresolved_alerts=unresolved_count,
        calls_per_day=calls_per_day,
        customer_emotion_distribution=emotion_distribution,
        top_agents_by_satisfaction=top_satisfaction,
        top_agents_by_anger=top_anger,
        recent_alerts=alert_summaries,
    )


# ── GET /admin/calls ─────────────────────────────────────────────────────────

@router.get("/calls", response_model=CallsListResponse)
async def list_all_calls(
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    agent_id: uuid.UUID | None = Query(None),
    status: str | None = Query(None, pattern="^(processing|done|error)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    _: Admin = Depends(require_admin),
):
    conditions = _date_conditions(start_date, end_date)
    if agent_id:
        conditions.append(Call.agent_id == agent_id)
    if status:
        conditions.append(Call.status == status)

    stmt = select(Call)
    if conditions:
        stmt = stmt.where(and_(*conditions))

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
        agent = await db.get(Agent, call.agent_id)
        summaries.append(CallSummary(
            call_id=call.id, agent_name=agent.name if agent else "Unknown",
            duration=call.duration,
            dominant_emotion=dominant_emotion(segs),
            alert_level=alert_level(bool(unresolved), anger_pct),
            status=call.status, start_time=call.start_time,
        ))

    return CallsListResponse(
        calls=summaries, total=total, page=page,
        page_size=page_size, total_pages=_pages(total, page_size),
    )
