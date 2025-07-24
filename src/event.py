from pydantic import BaseModel

from .action import Decision


class DecisionEvent(BaseModel):
    agent_id: str
    round: int
    decision: Decision


class StartTurnEvent(BaseModel):
    agent_id: str
    round: int


class DeathEvent(BaseModel):
    agent_id: str
    round: int


EventUnion = DecisionEvent | StartTurnEvent | DeathEvent
