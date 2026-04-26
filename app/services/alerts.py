"""
Alert detection logic run after a call is fully processed.
"""
from datetime import datetime


POSITIVE_EMOTIONS = {"happy", "neutral", "surprise"}
ANGER_SUSTAINED_THRESHOLD = 40   # 40 × 3 s = 120 s = 2 minutes


def check_alerts(
    call_id: str,
    customer_timeline: list[dict],
) -> list[dict]:
    """
    Analyse the customer emotion timeline and return a list of alert dicts.

    Each dict contains:
        type        – 'anger_sustained' | 'escalation'
        description – human-readable explanation
    """
    alerts = []

    # ── Alert 1: sustained anger (2+ consecutive minutes) ────────────────────
    angry_streak = 0
    for seg in customer_timeline:
        if seg["emotion"] == "angry" and seg["confidence"] > 0.5:
            angry_streak += 1
            if angry_streak >= ANGER_SUSTAINED_THRESHOLD:
                alerts.append({
                    "type": "anger_sustained",
                    "description": "Customer was angry for 2+ consecutive minutes",
                })
                break
        else:
            angry_streak = 0

    # ── Alert 2: escalation pattern (neutral → sad → angry) ──────────────────
    emotions = [s["emotion"] for s in customer_timeline]
    for i in range(len(emotions) - 2):
        if (
            emotions[i] == "neutral"
            and emotions[i + 1] == "sad"
            and emotions[i + 2] == "angry"
        ):
            alerts.append({
                "type": "escalation",
                "description": "Escalation pattern detected: neutral → sad → angry",
            })
            break

    return alerts


def dominant_emotion(segments: list) -> str | None:
    """
    Return the most frequent emotion across a list of EmotionSegment ORM objects.
    """
    if not segments:
        return None
    counts: dict[str, int] = {}
    for seg in segments:
        counts[seg.emotion] = counts.get(seg.emotion, 0) + 1
    return max(counts, key=lambda e: counts[e])


def alert_level(has_unresolved_alerts: bool, anger_pct: float) -> str:
    """
    Derive a traffic-light level for the calls list view.
      red    – unresolved alert exists
      yellow – no alert but anger > 20 %
      green  – calm call
    """
    if has_unresolved_alerts:
        return "red"
    if anger_pct > 0.20:
        return "yellow"
    return "green"
