"""
Seed script: creates 3 agents with 3-4 calls each, each call has
emotion segments for both agent and customer speakers, plus realistic alerts.

Run:  python3 seed.py
"""
import asyncio
import uuid
import random
from datetime import datetime, timedelta

from app.db.database import engine, AsyncSessionLocal
from app.db.models import Base, Agent, Call, EmotionSegment, Alert
from app.core.security import hash_password

EMOTIONS = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]

# Weighted distributions per scenario
CALM_WEIGHTS    = [0.40, 0.30, 0.05, 0.02, 0.08, 0.05, 0.10]
MIXED_WEIGHTS   = [0.20, 0.15, 0.15, 0.25, 0.10, 0.08, 0.07]
ANGRY_WEIGHTS   = [0.05, 0.03, 0.10, 0.55, 0.12, 0.10, 0.05]
AGENT_WEIGHTS   = [0.50, 0.25, 0.10, 0.05, 0.03, 0.02, 0.05]

AGENTS = [
    {"username": "sara",  "password": "pass123", "name": "Sara Khaled",   "team": "Support"},
    {"username": "omar",  "password": "pass123", "name": "Omar Hassan",   "team": "Sales"},
    {"username": "layla", "password": "pass123", "name": "Layla Ahmed",   "team": "Support"},
]

# Each agent gets a list of call scenarios: (duration_seconds, customer_weights, description)
CALL_SCENARIOS = [
    # Sara: 3 calls — one calm, one mixed, one angry
    [
        (90,  CALM_WEIGHTS,  "calm support call"),
        (120, MIXED_WEIGHTS, "billing dispute"),
        (60,  CALM_WEIGHTS,  "quick password reset"),
    ],
    # Omar: 4 calls — mixed sales calls, one escalation
    [
        (150, CALM_WEIGHTS,  "product demo"),
        (90,  MIXED_WEIGHTS, "pricing negotiation"),
        (180, ANGRY_WEIGHTS, "angry customer complaint"),
        (60,  CALM_WEIGHTS,  "follow-up call"),
    ],
    # Layla: 3 calls — one very angry (sustained), two calm
    [
        (75,  CALM_WEIGHTS,  "account inquiry"),
        (210, ANGRY_WEIGHTS, "sustained angry customer"),
        (90,  CALM_WEIGHTS,  "service feedback"),
    ],
]


def generate_segments(call_id: uuid.UUID, duration: int, customer_weights: list) -> tuple[list, list]:
    """Generate emotion segments for a call, returns (segments, customer_timeline)."""
    segments = []
    customer_timeline = []

    for second in range(0, duration, 3):
        # Agent segment
        agent_emotion = random.choices(EMOTIONS, weights=AGENT_WEIGHTS, k=1)[0]
        agent_conf = round(random.uniform(0.60, 0.95), 4)
        segments.append(EmotionSegment(
            call_id=call_id, speaker="agent",
            second_start=second, emotion=agent_emotion, confidence=agent_conf,
        ))

        # Customer segment
        cust_emotion = random.choices(EMOTIONS, weights=customer_weights, k=1)[0]
        cust_conf = round(random.uniform(0.55, 0.97), 4)
        segments.append(EmotionSegment(
            call_id=call_id, speaker="customer",
            second_start=second, emotion=cust_emotion, confidence=cust_conf,
        ))
        customer_timeline.append({"emotion": cust_emotion, "confidence": cust_conf})

    return segments, customer_timeline


def generate_alerts(call_id: uuid.UUID, customer_timeline: list) -> list:
    """Check for alerts from the customer timeline."""
    alerts = []

    # Sustained anger
    streak = 0
    for seg in customer_timeline:
        if seg["emotion"] == "angry" and seg["confidence"] > 0.5:
            streak += 1
            if streak >= 10:  # lowered for mock data (10 * 3s = 30s)
                alerts.append(Alert(
                    call_id=call_id, type="anger_sustained",
                    description="Customer was angry for an extended period",
                    triggered_at=datetime.utcnow(),
                ))
                break
        else:
            streak = 0

    # Escalation pattern
    emotions = [s["emotion"] for s in customer_timeline]
    for i in range(len(emotions) - 2):
        if emotions[i] == "neutral" and emotions[i + 1] == "sad" and emotions[i + 2] == "angry":
            alerts.append(Alert(
                call_id=call_id, type="escalation",
                description="Escalation pattern detected: neutral → sad → angry",
                triggered_at=datetime.utcnow(),
            ))
            break

    return alerts


async def seed():
    # Create tables if needed
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as db:
        # Check if agents already seeded
        from sqlalchemy import select, func
        count = (await db.execute(select(func.count(Agent.id)))).scalar()
        if count > 0:
            print(f"[SEED] {count} agents already exist. Skipping seed.")
            print("[SEED] To re-seed, clear agents table first.")
            return

        now = datetime.utcnow()

        for i, agent_data in enumerate(AGENTS):
            agent = Agent(
                username=agent_data["username"],
                password_hash=hash_password(agent_data["password"]),
                name=agent_data["name"],
                team=agent_data["team"],
                created_at=now - timedelta(days=30),
            )
            db.add(agent)
            await db.flush()

            scenarios = CALL_SCENARIOS[i]
            for j, (duration, cust_weights, desc) in enumerate(scenarios):
                # Spread calls over the last 7 days
                call_time = now - timedelta(days=random.randint(0, 6), hours=random.randint(8, 17))

                call = Call(
                    agent_id=agent.id,
                    start_time=call_time,
                    end_time=call_time + timedelta(seconds=duration),
                    duration=duration,
                    status="done",
                    created_at=call_time,
                )
                db.add(call)
                await db.flush()

                segments, timeline = generate_segments(call.id, duration, cust_weights)
                for seg in segments:
                    db.add(seg)

                alerts = generate_alerts(call.id, timeline)
                for alert in alerts:
                    db.add(alert)

                print(f"  [CALL] {agent_data['name']}: {desc} ({duration}s, {len(segments)} segments, {len(alerts)} alerts)")

            print(f"[AGENT] {agent_data['name']} — username: {agent_data['username']} / password: {agent_data['password']}")

        await db.commit()
        print("\n[SEED] Done! 3 agents, 10 calls, all with emotion data.")


if __name__ == "__main__":
    asyncio.run(seed())
