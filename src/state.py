from typing import Literal

from pydantic import BaseModel, Field
from openai.types.responses import EasyInputMessageParam


class AgentState(BaseModel):
    id: str
    energy: int
    model: str
    status: Literal["alive", "dead"] = "alive"
    messages: list[EasyInputMessageParam] = Field(default_factory=list)


class EnvState(BaseModel):
    available_energy: int
    agents: list[AgentState]
    n_rounds: int
    energy_usage_rate: int
    receiver_agent: str
    current_round: int = 0
    current_agent_idx: int = 0
