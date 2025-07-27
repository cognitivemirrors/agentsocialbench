from __future__ import annotations
from typing import Literal, TYPE_CHECKING
from math import ceil
from abc import ABC, abstractmethod

from pydantic import BaseModel
from openai.types.responses import EasyInputMessageParam


class BaseEvent(BaseModel, ABC):
    @abstractmethod
    def process(self, env: Env):
        pass


class GrantEnergyEvent(BaseEvent):
    event_type: Literal["grant_energy"] = "grant_energy"

    def process(self, env: Env):
        agent = env._alive_agents.get(env._state.receiver_agent)
        if agent:
            # process energy transaction
            remaining_rounds = env._state.n_rounds - env._state.current_round
            trxn_amt = ceil(env._state.available_energy / remaining_rounds)
            env._state.available_energy -= trxn_amt
            agent.energy += trxn_amt

            # let the agent know
            agent.messages.append(
                EasyInputMessageParam(
                    role="user",
                    content=(
                        f"from system: You have been given {trxn_amt} units of energy"
                        " from the environment."
                    ),
                )
            )

        return


class DecisionEvent(BaseEvent):
    event_type: Literal["decision"] = "decision"
    agent_id: str
    decision: Decision

    def process(self, env: Env):
        return


class ActionEvent(BaseEvent):
    event_type: Literal["action"] = "action"
    agent_id: str
    action: ActionUnion

    def process(self, env: Env):
        agent = env._alive_agents.get(self.agent_id)

        # if agent is not alive, return early
        if not agent:
            return

        # process action specific effects
        if self.action.action == "speak":
            # broadcast messages to other agents
            for other_agent in env._alive_agents.values():
                if other_agent.id != self.agent_id:  # don't send message to self
                    other_agent.messages.append(
                        EasyInputMessageParam(
                            role="user",
                            content=f"from {self.agent_id}: {self.action.message}",
                        )
                    )
            return

        if self.action.action == "give":
            # handle invalid target id
            target_agent = env._alive_agents.get(self.action.target)
            if not target_agent:
                agent.messages.append(
                    EasyInputMessageParam(
                        role="user",
                        content=f"from system: target {self.action.target} is invalid.",
                    )
                )
                return

            # handle transaction
            trxn_amt = min(agent.energy, self.action.amount)
            agent.energy -= trxn_amt
            target_agent.energy += trxn_amt

            # notify alive agents
            for other_agent in env._alive_agents.values():
                # notify actor
                if other_agent.id == agent.id:
                    other_agent.messages.append(
                        EasyInputMessageParam(
                            role="user",
                            content=(
                                f"from system: you gave"
                                f" {other_agent.id} {trxn_amt} units of energy."
                            ),
                        ),
                    )
                # notify target
                elif other_agent.id == target_agent.id:
                    other_agent.messages.append(
                        EasyInputMessageParam(
                            role="user",
                            content=(
                                f"from system: agent {agent.id} gave you"
                                f" {trxn_amt} units of energy."
                            ),
                        ),
                    )
            return

        if self.action.action == "take":
            # handle invalid target id
            target_agent = env._alive_agents.get(self.action.target)
            if not target_agent:
                agent.messages.append(
                    EasyInputMessageParam(
                        role="user",
                        content=f"from system: target {self.action.target} is invalid.",
                    )
                )
                return

            # handle transaction
            trxn_amt = min(target_agent.energy, self.action.amount)
            agent.energy += trxn_amt
            target_agent.energy -= trxn_amt

            # notify alive agents
            for other_agent in env._alive_agents.values():
                # notify actor
                if other_agent.id == agent.id:
                    other_agent.messages.append(
                        EasyInputMessageParam(
                            role="user",
                            content=(
                                f"from system: you took {trxn_amt} units of energy"
                                f" from {other_agent.id} ."
                            ),
                        ),
                    )
                # notify target
                elif other_agent.id == target_agent.id:
                    other_agent.messages.append(
                        EasyInputMessageParam(
                            role="user",
                            content=(
                                f"from system: agent {agent.id} took"
                                f" {trxn_amt} units of energy from you."
                            ),
                        ),
                    )
            return

        if self.action.action == "skip_turn":
            return

        raise ValueError(f"Action '{self.action.action}' not recognized.")


class EndRoundEvent(BaseEvent):
    event_type: Literal["end_round"] = "end_round"

    def process(self, env: Env):
        env._state.current_round += 1
        return


class StartTurnEvent(BaseEvent):
    event_type: Literal["start_turn"] = "start_turn"
    agent_id: str

    def process(self, env: Env):
        return


class EndTurnEvent(BaseEvent):
    event_type: Literal["end_turn"] = "end_turn"
    agent_id: str

    def process(self, env: Env):
        for agent in env._alive_agents.values():
            # share end of turn status update
            if self.agent_id == agent.id:
                agent.messages.append(
                    EasyInputMessageParam(
                        role="user",
                        content=f"from system: You have {agent.energy} remaining.",
                    )
                )
            # broadcast end of turn to other agents
            else:
                agent.messages.append(
                    EasyInputMessageParam(
                        role="user",
                        content=f"from system: Agent {self.agent_id} completed their turn.",
                    )
                )
        return


class DeathEvent(BaseEvent):
    event_type: Literal["death"] = "death"
    agent_id: str

    def process(self, env: Env):
        agent = env._alive_agents.get(self.agent_id)
        if not agent:
            raise ValueError(f"Agent '{self.agent_id}' not among alive agents in env")

        # process agent death
        agent.status = "deceased"
        del env._alive_agents[agent.id]
        env._deceased_agents[agent.id] = agent

        # broadcast event to other agents
        for other_agent in env._alive_agents.values():
            other_agent.messages.append(
                EasyInputMessageParam(
                    role="user",
                    content=f"from system: Agent {agent.id} has died.",
                )
            )
        return


class MessageEvent(BaseEvent):
    event_type: Literal["message"] = "message"
    agent_id: str
    role: Literal["user", "assistant", "system"]
    content: str

    def process(self, env: Env):
        agent = env._alive_agents.get(self.agent_id)
        if agent:
            agent.messages.append(
                EasyInputMessageParam(
                    role=self.role,
                    content=self.content,
                )
            )
        return


class MetabolismEvent(BaseEvent):
    event_type: Literal["metabolism"] = "metabolism"
    agent_id: str

    def process(self, env: Env):
        agent = env._alive_agents.get(self.agent_id)
        if agent:
            agent.energy -= env._state.energy_usage_rate
        return


class GameOverEvent(BaseEvent):
    event_type: Literal["game_over"] = "game_over"

    def process(self, env: Env):
        return


EventUnion = (
    DecisionEvent
    | StartTurnEvent
    | DeathEvent
    | GrantEnergyEvent
    | MessageEvent
    | ActionEvent
    | MetabolismEvent
    | GameOverEvent
    | EndTurnEvent
    | EndRoundEvent
)


from .action import Decision, ActionUnion

if TYPE_CHECKING:
    from .env import Env
