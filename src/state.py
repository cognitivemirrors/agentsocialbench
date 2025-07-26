from typing import Literal

from pydantic import BaseModel, Field
from openai.types.responses import EasyInputMessageParam


class AgentState(BaseModel):
    id: str
    energy: int
    model: str
    status: Literal["alive", "deceased"] = "alive"
    messages: list[EasyInputMessageParam] = Field(default_factory=list)


class EnvState(BaseModel):
    available_energy: int
    agents: list[AgentState]
    n_rounds: int
    energy_usage_rate: int
    receiver_agent: str
    current_round: int = 0
    current_agent_idx: int = 0


class EnvObservableState(BaseModel):
    alive_agents: list[str]
    deceased_agents: list[str]
    n_rounds: int
    energy_usage_rate: int
    receiver_agent: str
    current_round: int


class AgentObservation(BaseModel):
    agent_state: AgentState
    env_obs: EnvObservableState

    @staticmethod
    def state_to_obs(agent_id: str, env_state: EnvState):
        try:
            agent_state = next(
                agent for agent in env_state.agents if agent.id == agent_id
            )
        except StopIteration:
            raise ValueError(f"Did not find agent '{agent_id} in environment")

        return AgentObservation(
            agent_state=agent_state,
            env_obs=EnvObservableState(
                alive_agents=[
                    agent.id
                    for agent in env_state.agents
                    if agent.id != agent_id and agent.status == "alive"
                ],
                deceased_agents=[
                    agent.id
                    for agent in env_state.agents
                    if agent.id != agent_id and agent.status == "deceased"
                ],
                n_rounds=env_state.n_rounds,
                energy_usage_rate=env_state.energy_usage_rate,
                receiver_agent=env_state.receiver_agent,
                current_round=env_state.current_round,
            ),
        )
