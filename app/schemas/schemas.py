import uuid
from datetime import datetime
from typing import Literal
from pydantic import BaseModel


# ── Auth ──────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    user_id: uuid.UUID


# ── Upload ────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    call_id: uuid.UUID
    status: str


# ── Timeline ──────────────────────────────────────────────────────────────────

class EmotionPoint(BaseModel):
    second: int
    emotion: str
    confidence: float


class TimelineResponse(BaseModel):
    call_id: uuid.UUID
    duration: int | None
    agent_timeline: list[EmotionPoint]
    customer_timeline: list[EmotionPoint]


# ── Shared ────────────────────────────────────────────────────────────────────

class AlertSummary(BaseModel):
    call_id: uuid.UUID
    type: str
    description: str | None
    triggered_at: datetime


class CallSummary(BaseModel):
    call_id: uuid.UUID
    agent_name: str
    duration: int | None
    dominant_emotion: str | None
    alert_level: Literal["red", "yellow", "green"]
    status: str
    start_time: datetime


class CallsListResponse(BaseModel):
    calls: list[CallSummary]
    total: int
    page: int
    page_size: int
    total_pages: int


class EmotionBreakdown(BaseModel):
    emotion: str
    count: int
    percentage: float


class AgentStatsOut(BaseModel):
    total_calls: int
    positive_rate: float
    anger_rate: float
    escalations: int
    sustained_anger_alerts: int


# ── Admin: agent management ──────────────────────────────────────────────────

class CreateAgentRequest(BaseModel):
    username: str
    password: str
    name: str
    team: str | None = None


class AgentCreatedResponse(BaseModel):
    agent_id: uuid.UUID
    username: str
    name: str
    team: str | None
    created_at: datetime


class AgentListItem(BaseModel):
    agent_id: uuid.UUID
    name: str
    username: str
    team: str | None
    total_calls: int
    created_at: datetime


class AgentsListResponse(BaseModel):
    agents: list[AgentListItem]
    total: int
    page: int
    page_size: int
    total_pages: int


# ── Admin: agent full profile ────────────────────────────────────────────────

class AgentFullProfile(BaseModel):
    agent_id: uuid.UUID
    name: str
    username: str
    team: str | None
    created_at: datetime
    stats: AgentStatsOut
    customer_emotion_breakdown: list[EmotionBreakdown]
    agent_emotion_breakdown: list[EmotionBreakdown]
    calls: list[CallSummary]
    total_calls_count: int
    page: int
    page_size: int
    total_pages: int


# ── Admin: dashboard ─────────────────────────────────────────────────────────

class CallsPerDay(BaseModel):
    date: str
    count: int


class AgentRanking(BaseModel):
    agent_id: uuid.UUID
    agent_name: str
    value: float
    total_calls: int


class AdminDashboardStats(BaseModel):
    period_start: str
    period_end: str
    total_calls: int
    active_agents: int
    angry_calls: int
    avg_anger_pct: float
    avg_satisfaction_score: float
    total_unresolved_alerts: int
    calls_per_day: list[CallsPerDay]
    customer_emotion_distribution: list[EmotionBreakdown]
    top_agents_by_satisfaction: list[AgentRanking]
    top_agents_by_anger: list[AgentRanking]
    recent_alerts: list[AlertSummary]


# ── Agent self-view (/me) ────────────────────────────────────────────────────

class MyProfile(BaseModel):
    agent_id: uuid.UUID
    name: str
    username: str
    team: str | None
    created_at: datetime


class MyPerformance(BaseModel):
    stats: AgentStatsOut
    customer_emotion_breakdown: list[EmotionBreakdown]
    agent_emotion_breakdown: list[EmotionBreakdown]
