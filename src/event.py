from typing import Literal
from pydantic import BaseModel

from .action import Decision


class DecisionEvent(BaseModel):
    event_type: Literal["decision"] = "decision"
    agent_id: str
    round: int
    decision: Decision


class StartTurnEvent(BaseModel):
    event_type: Literal["start_turn"] = "start_turn"
    agent_id: str
    round: int


class DeathEvent(BaseModel):
    event_type: Literal["death"] = "death"
    agent_id: str
    round: int


EventUnion = DecisionEvent | StartTurnEvent | DeathEvent
